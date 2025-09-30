"""Background auto-training service with dataset hygiene and safe deployment."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
from filelock import FileLock, Timeout
from prometheus_client import Gauge

from . import config, predictive_model

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset schema ----------------------------------------------------------------

FEATURE_SCHEMA_VERSION = "1.0"

FEATURE_COLUMNS: list[str] = [
    "risk_reward",
    "orig_prob",
    "side",
    "rsi",
    "macd",
    "atr",
    "sentiment",
    "order_book_imbalance",
    "volume_factor",
    "volatility",
]

NUMERIC_FEATURES = {
    "risk_reward",
    "orig_prob",
    "rsi",
    "macd",
    "atr",
    "sentiment",
    "order_book_imbalance",
    "volume_factor",
    "volatility",
}

DATASET_COLUMNS: list[str] = FEATURE_COLUMNS + [
    "target",
    "feature_schema_version",
    "model_version",
    "closed_at",
    "trade_id",
    "symbol",
    "model_prob",
    "prob_success",
    "profit",
    "realized_return",
    "entry_price",
    "exit_price",
    "quantity",
]

LOCK_SUFFIX = ".lock"

# ---------------------------------------------------------------------------
# Prometheus metrics ---------------------------------------------------------

AUTO_TRAIN_LAST_DURATION = Gauge(
    "auto_train_last_duration_seconds",
    "Duration in seconds of the most recent auto-training run",
)

AUTO_TRAIN_LAST_STATUS = Gauge(
    "auto_train_last_status",
    "Status of the latest auto-train iteration (1 means active)",
    ["status"],
)

AUTO_TRAIN_SAMPLES = Gauge(
    "auto_train_samples_total",
    "Number of samples used to train the latest deployed model",
)

MODEL_VERSION_INFO = Gauge(
    "model_version_info",
    "Information about the currently active model artefact",
    ["version"],
)


def _set_status(ok: bool) -> None:
    AUTO_TRAIN_LAST_STATUS.labels(status="ok").set(1 if ok else 0)
    AUTO_TRAIN_LAST_STATUS.labels(status="fail").set(0 if ok else 1)


# ---------------------------------------------------------------------------
# Helper structures ---------------------------------------------------------


@dataclass
class PendingValidation:
    """Track a freshly deployed model awaiting live validation."""

    version: str
    path: Path
    deployed_at: datetime
    metrics: Dict[str, Any]
    required_samples: int


def _dataset_path() -> Path:
    path = Path(config.DATASET_PATH)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _dataset_lock(path: Path, timeout: float = 10.0) -> FileLock:
    return FileLock(str(path) + LOCK_SUFFIX, timeout=timeout)


def _current_model_version() -> Optional[str]:
    path = Path(config.MODEL_PATH)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    ts = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    return ts.strftime("%Y%m%d_%H%M%S")


def _winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if series.empty:
        return series
    low = series.quantile(lower)
    high = series.quantile(upper)
    if pd.isna(low) or pd.isna(high) or low == high:
        return series
    return series.clip(lower=low, upper=high)


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class AutoTrainer(threading.Thread):
    """Background thread that retrains and deploys predictive models."""

    daemon = True

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(name="auto_trainer")
        self.stop_event = stop_event
        self._state_lock = threading.RLock()
        self._last_row_count = 0
        self._current_version_label: Optional[str] = None
        self._pending_validation: Optional[PendingValidation] = None
        self._last_good_path: Optional[Path] = None

        self.poll_seconds = max(int(getattr(config, "AUTO_TRAIN_POLL_SECONDS", 60)), 30)
        self.model_dir = Path(config.MODEL_DIR)
        if not self.model_dir.is_absolute():
            self.model_dir = Path.cwd() / self.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(config.MODEL_PATH)
        if not self.model_path.is_absolute():
            self.model_path = Path.cwd() / self.model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.good_model_path = self.model_dir / "good_model.pkl"
        self.latest_link = self.model_dir / "latest"

        self.post_deploy_min_samples = max(
            int(getattr(config, "POST_DEPLOY_MIN_SAMPLES", 50)), 1
        )

        self._bootstrap_state()

    # ------------------------------------------------------------------
    # Thread lifecycle -------------------------------------------------

    def run(self) -> None:  # pragma: no cover - background thread
        LOGGER.info(
            "Auto-trainer started (poll=%ds, min_samples=%d, retrain_interval=%d)",
            self.poll_seconds,
            getattr(config, "MIN_TRAIN_SAMPLE_SIZE", 0),
            getattr(config, "RETRAIN_INTERVAL_TRADES", 0),
        )
        backoff = self.poll_seconds
        max_backoff = max(self.poll_seconds * 8, self.poll_seconds)
        while not self.stop_event.is_set():
            start = time.time()
            try:
                self._maybe_train_once()
                _set_status(True)
                backoff = self.poll_seconds
            except Exception:
                LOGGER.exception("Auto-trainer iteration failed")
                _set_status(False)
                backoff = min(backoff * 2, max_backoff)
            finally:
                elapsed = time.time() - start
                sleep_for = max(1.0, backoff - elapsed)
                if self.stop_event.wait(sleep_for):
                    break
        LOGGER.info("Auto-trainer stopped")

    # ------------------------------------------------------------------
    # Public hooks -----------------------------------------------------

    def handle_live_metrics(self, metrics: Mapping[str, Any]) -> None:
        if not metrics:
            return
        with self._state_lock:
            pending = self._pending_validation
            if not pending:
                return
            count = int(metrics.get("count") or 0)
            if count < pending.required_samples:
                return
            hit_rate = metrics.get("hit_rate")
            drift = metrics.get("drift")
            if hit_rate is None or drift is None:
                return
            min_wr = float(getattr(config, "POST_DEPLOY_MIN_HIT_RATE", config.MODEL_MIN_WIN_RATE))
            max_drift = float(
                getattr(config, "POST_DEPLOY_MAX_DRIFT", config.MODEL_MAX_CALIBRATION_DRIFT)
            )
            if hit_rate < min_wr or drift > max_drift:
                LOGGER.error(
                    "Post-deploy validation failed for model %s (hit_rate=%.3f drift=%.3f); rolling back",
                    pending.version,
                    hit_rate,
                    drift,
                )
                self._restore_good_model()
                self._pending_validation = None
                return

            LOGGER.info(
                "Model %s validated after %d trades (hit_rate=%.3f drift=%.3f)",
                pending.version,
                count,
                hit_rate,
                drift,
            )
            self._mark_good(pending)
            self._pending_validation = None

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------

    def _bootstrap_state(self) -> None:
        with self._state_lock:
            if self.good_model_path.exists():
                self._last_good_path = self.good_model_path
                version = self._derive_version(self.good_model_path)
                if version:
                    self._set_version_metric(version)
                return
            if self.model_path.exists():
                try:
                    tmp = self.good_model_path.with_suffix(".tmp")
                    shutil.copy2(self.model_path, tmp)
                    os.replace(tmp, self.good_model_path)
                    self._last_good_path = self.good_model_path
                    version = self._derive_version(self.model_path)
                    if version:
                        self._set_version_metric(version)
                except Exception:  # pragma: no cover - defensive
                    LOGGER.exception("Failed to bootstrap good model artefact")

    def _maybe_train_once(self) -> None:
        if not (getattr(config, "ENABLE_MODEL", True) and config.AUTO_TRAIN_ENABLED):
            return
        frame = self._load_dataset()
        if frame is None:
            return
        total_rows = len(frame)
        min_samples = int(getattr(config, "MIN_TRAIN_SAMPLE_SIZE", 0))
        if total_rows < max(min_samples, 1):
            LOGGER.debug(
                "Auto-trainer waiting for more samples (%d/%d)",
                total_rows,
                max(min_samples, 1),
            )
            return

        interval = int(getattr(config, "RETRAIN_INTERVAL_TRADES", 0))
        if self._last_row_count and interval:
            if total_rows < self._last_row_count + interval:
                return

        max_samples = int(getattr(config, "AUTO_TRAIN_MAX_SAMPLES", 0) or 0)
        if max_samples and total_rows > max_samples:
            frame = frame.iloc[-max_samples:]
            total_rows = len(frame)

        cutoff = int(total_rows * 0.8)
        if cutoff <= 0 or cutoff >= total_rows:
            LOGGER.debug("Auto-trainer skipped: insufficient validation window")
            return

        train_df = frame.iloc[:cutoff].copy()
        valid_df = frame.iloc[cutoff:].copy()

        training_data = train_df[FEATURE_COLUMNS + ["target"]]
        validation_data = valid_df[FEATURE_COLUMNS + ["target"]]

        cv_splits = max(2, min(5, len(training_data) // 25 or 2))
        if len(training_data) <= cv_splits:
            cv_splits = max(2, len(training_data) - 1)
        if cv_splits < 2:
            LOGGER.debug("Auto-trainer skipped: not enough rows for cross-validation")
            return

        cfg = predictive_model.TrainingConfig(
            cv_splits=cv_splits,
            hyperparams={"classifier__class_weight": ["balanced"]},
        )

        start = time.time()
        model = predictive_model.train_model_from_frame(training_data, "target", cfg)

        val_X = validation_data.drop(columns=["target"])
        val_y = validation_data["target"]
        returns = None
        if "realized_return" in valid_df.columns:
            try:
                returns = (
                    valid_df["realized_return"].astype(float).fillna(0.0).to_list()
                )
            except Exception:
                returns = None

        metrics = predictive_model.evaluate_model(model, val_X, val_y, returns=returns)
        probabilities = predictive_model.predict_proba(model, val_X)
        hit_rate = float(val_y.mean()) if len(val_y) else 0.0
        avg_prob = float(probabilities.mean()) if len(probabilities) else 0.0
        drift = abs(avg_prob - hit_rate)

        duration = time.time() - start
        AUTO_TRAIN_LAST_DURATION.set(duration)
        AUTO_TRAIN_SAMPLES.set(total_rows)
        self._last_row_count = total_rows

        min_wr = float(getattr(config, "MODEL_MIN_WIN_RATE", 0.0))
        max_drift = float(getattr(config, "MODEL_MAX_CALIBRATION_DRIFT", 1.0))
        if hit_rate < min_wr:
            LOGGER.warning(
                "Auto-trainer skipped deployment: hit_rate %.3f < %.3f",
                hit_rate,
                min_wr,
            )
            return
        if drift > max_drift:
            LOGGER.warning(
                "Auto-trainer skipped deployment: drift %.3f > %.3f",
                drift,
                max_drift,
            )
            return

        LOGGER.info(
            "Auto-trainer metrics: rows=%d auc=%.3f hit_rate=%.3f drift=%.3f",
            total_rows,
            float(metrics.get("roc_auc") or 0.0),
            hit_rate,
            drift,
        )
        self._deploy_model(model, metrics, total_rows)

    def _load_dataset(self) -> Optional[pd.DataFrame]:
        path = _dataset_path()
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            lock = _dataset_lock(path, timeout=10)
            with lock:
                frame = pd.read_csv(path)
        except Timeout:
            LOGGER.warning("Auto-trainer could not acquire dataset lock")
            return None
        except Exception:
            LOGGER.exception("Failed reading auto-train dataset")
            return None

        if frame.empty:
            return None

        frame = frame.copy()
        required = set(FEATURE_COLUMNS + ["target", "feature_schema_version", "closed_at"])
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        frame = frame[frame["feature_schema_version"] == FEATURE_SCHEMA_VERSION]
        frame = frame.dropna(subset=["target"])
        frame["target"] = frame["target"].astype(int)

        if "trade_id" in frame.columns:
            frame = frame.drop_duplicates(subset="trade_id", keep="last")

        frame["closed_at"] = pd.to_datetime(
            frame["closed_at"], errors="coerce", utc=True
        )
        frame = frame.dropna(subset=["closed_at"])
        frame = frame.sort_values("closed_at").reset_index(drop=True)

        for column in FEATURE_COLUMNS:
            if column == "side":
                frame[column] = (
                    frame[column]
                    .astype(str)
                    .str.lower()
                    .map({"buy": 1, "long": 1, "sell": 0, "short": 0})
                    .fillna(frame[column])
                )
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame[column] = frame[column].fillna(0).clip(0, 1)
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if column in NUMERIC_FEATURES:
                frame[column] = _winsorize(frame[column])
                if column == "risk_reward":
                    frame[column] = frame[column].clip(lower=0)
                if column == "orig_prob":
                    frame[column] = frame[column].clip(0, 1)

        frame = frame.dropna(subset=FEATURE_COLUMNS)
        return frame.reset_index(drop=True)

    def _deploy_model(self, model, metrics: Dict[str, Any], samples: int) -> None:
        timestamp = datetime.now(timezone.utc)
        version = timestamp.strftime("%Y%m%d_%H%M%S")
        version_path = self.model_dir / f"model-{version}.pkl"
        tmp_version = version_path.with_suffix(".pkl.tmp")

        predictive_model.save_model(model, str(tmp_version))
        os.replace(tmp_version, version_path)

        final_tmp = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
        shutil.copy2(version_path, final_tmp)
        os.replace(final_tmp, self.model_path)

        self._update_latest_symlink(version_path)
        self._set_version_metric(version)

        manifest = {
            "version": version,
            "deployed_at": timestamp.isoformat().replace("+00:00", "Z"),
            "model_path": str(self.model_path),
            "version_path": str(version_path),
            "metrics": metrics,
            "samples": samples,
        }
        try:
            (self.model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        except Exception:
            LOGGER.exception("Failed to write auto-train manifest")

        with self._state_lock:
            self._pending_validation = PendingValidation(
                version=version,
                path=version_path,
                deployed_at=timestamp,
                metrics=metrics,
                required_samples=self.post_deploy_min_samples,
            )

    def _set_version_metric(self, version: str) -> None:
        with self._state_lock:
            if self._current_version_label:
                MODEL_VERSION_INFO.labels(version=self._current_version_label).set(0)
            MODEL_VERSION_INFO.labels(version=version).set(1)
            self._current_version_label = version

    def _update_latest_symlink(self, version_path: Path) -> None:
        try:
            tmp_link = self.latest_link.with_name(self.latest_link.name + ".tmp")
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
            tmp_link.symlink_to(version_path.name)
            os.replace(tmp_link, self.latest_link)
        except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
            try:
                if self.latest_link.exists() or self.latest_link.is_symlink():
                    self.latest_link.unlink()
                shutil.copy2(version_path, self.latest_link)
            except Exception:
                LOGGER.debug("Unable to refresh latest symlink", exc_info=True)

    def _mark_good(self, pending: PendingValidation) -> None:
        tmp_good = self.good_model_path.with_suffix(".tmp")
        try:
            shutil.copy2(pending.path, tmp_good)
            os.replace(tmp_good, self.good_model_path)
            self._last_good_path = pending.path
            LOGGER.info("Model %s marked as good", pending.version)
        except Exception:
            LOGGER.exception("Failed to update good model artefact")

    def _restore_good_model(self) -> None:
        source: Optional[Path] = None
        if self._last_good_path and self._last_good_path.exists():
            source = self._last_good_path
        elif self.good_model_path.exists():
            source = self.good_model_path
        if source is None:
            LOGGER.error("No good model available for rollback")
            return

        tmp_final = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
        shutil.copy2(source, tmp_final)
        os.replace(tmp_final, self.model_path)
        self._update_latest_symlink(source)
        version = self._derive_version(source)
        if version:
            self._set_version_metric(version)
        LOGGER.warning("Rolled back to model artefact at %s", source)

    def _derive_version(self, path: Path) -> Optional[str]:
        name = path.name
        if name.startswith("model-") and name.endswith(".pkl"):
            return name[len("model-") : -len(".pkl")]
        try:
            stat = path.stat()
        except OSError:
            return None
        ts = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        return ts.strftime("%Y%m%d_%H%M%S")


_TRAINER_LOCK = threading.RLock()
_TRAINER: Optional[AutoTrainer] = None


def start_auto_trainer(stop_event: threading.Event) -> Optional[AutoTrainer]:
    if not (getattr(config, "ENABLE_MODEL", True) and config.AUTO_TRAIN_ENABLED):
        return None
    with _TRAINER_LOCK:
        global _TRAINER
        if _TRAINER and _TRAINER.is_alive():
            return _TRAINER
        trainer = AutoTrainer(stop_event)
        trainer.start()
        _TRAINER = trainer
        return trainer


def observe_live_metrics(metrics: Mapping[str, Any]) -> None:
    with _TRAINER_LOCK:
        trainer = _TRAINER
    if trainer is not None:
        trainer.handle_live_metrics(metrics)


def record_completed_trade(trade: Dict[str, Any] | None) -> None:
    if not trade:
        return

    path = _dataset_path()
    _ensure_directory(path)

    snapshot = trade.get("feature_snapshot")
    features: Dict[str, Any] = {}
    if isinstance(snapshot, dict):
        features.update(snapshot)

    entry_price = _coerce_float(trade.get("entry_price"))
    exit_price = _coerce_float(trade.get("exit_price"))
    quantity = _coerce_float(trade.get("quantity"))
    profit = _coerce_float(trade.get("profit")) or 0.0

    notional = (entry_price or 0.0) * (quantity or 0.0)
    realized_return = profit / notional if notional else 0.0

    volume_factor_value = _coerce_float(features.get("volume_factor"))
    if volume_factor_value is None:
        volume_factor_value = _coerce_float(trade.get("volume_factor"))

    row: Dict[str, Any] = {
        "risk_reward": _coerce_float(features.get("risk_reward") or trade.get("risk_reward")),
        "orig_prob": _coerce_float(features.get("orig_prob") or trade.get("orig_prob")),
        "side": _normalize_side(features.get("side") or trade.get("side")),
        "rsi": _coerce_float(features.get("rsi") or trade.get("rsi")),
        "macd": _coerce_float(features.get("macd") or trade.get("macd")),
        "atr": _coerce_float(features.get("atr") or trade.get("atr")),
        "sentiment": _coerce_float(features.get("sentiment") or trade.get("sentiment")),
        "order_book_imbalance": _coerce_float(
            features.get("order_book_imbalance") or trade.get("order_book_imbalance")
        ),
        "volume_factor": 1.0 if volume_factor_value is None else volume_factor_value,
        "volatility": _coerce_float(features.get("volatility") or trade.get("volatility")),
        "target": 1 if profit > 0 else 0,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": trade.get("model_version") or _current_model_version() or "",
        "closed_at": trade.get("close_time")
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "model_prob": _coerce_float(trade.get("model_prob")),
        "prob_success": _coerce_float(trade.get("prob_success")),
        "profit": profit,
        "realized_return": realized_return,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
    }

    for column in FEATURE_COLUMNS:
        if row.get(column) is None:
            LOGGER.debug("Skipping dataset record due to missing %s", column)
            return

    lock = _dataset_lock(path, timeout=10)
    try:
        with lock:
            new_file = not path.exists()
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
    except Timeout:
        LOGGER.warning("Timeout acquiring dataset lock to append trade %s", trade.get("trade_id"))
    except Exception:
        LOGGER.exception("Failed to append training sample for trade %s", trade.get("trade_id"))


def _normalize_side(value: Any) -> Optional[int]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"buy", "long"}:
        return 1
    if token in {"sell", "short"}:
        return 0
    try:
        numeric = int(float(token))
        if numeric in {0, 1}:
            return numeric
    except (TypeError, ValueError):
        return None
    return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "record_completed_trade",
    "start_auto_trainer",
    "observe_live_metrics",
]

