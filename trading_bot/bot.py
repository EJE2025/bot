import argparse
import json
import logging
import os

USE_EVENTLET = os.getenv("USE_EVENTLET", "1") == "1"
_EVENTLET_IMPORT_ERROR: str | None = None

if USE_EVENTLET:
    try:
        import eventlet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        eventlet = None  # type: ignore[assignment]
        USE_EVENTLET = False
        _EVENTLET_IMPORT_ERROR = str(exc)
    else:  # pragma: no cover - only executed when eventlet is installed
        eventlet.monkey_patch(
            os=True,
            select=True,
            socket=True,
            thread=True,
            time=True,
            subprocess=True,
        )

import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from threading import RLock, Thread
from typing import Any

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - optional dependency
    binomtest = None

from . import (
    config,
    auto_trainer,
    data,
    execution,
    strategy,
    predictive_model,
    notify,
    optimizer,
    permissions,
    trade_manager,
    shadow,
    shutdown,
    mode as bot_mode,
    exporter,
)
from . import webapp
from .trade_manager import (
    add_trade,
    close_trade,
    find_trade,
    update_trade,
    all_open_trades,
    load_trades,
    save_trades,
    count_open_trades,
    count_trades_for_symbol,
    set_trade_state,
)
from .reconcile import reconcile_pending_trades
from .state_machine import TradeState
from .metrics import (
    start_metrics_server,
    update_trade_metrics,
    record_model_performance,
    maybe_alert,
)
from .monitor import monitor_system
from .utils import normalize_symbol

if not config.BITGET_API_KEY:
    print("\u26a0\ufe0f  No se carg\xf3 la API KEY. Revisa si el archivo .env existe o el gestor de secretos est\xe1 configurado.")
else:
    print("\u2705 Archivo .env cargado correctamente.")

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

if globals().get("_EVENTLET_IMPORT_ERROR"):
    logger.warning(
        "USE_EVENTLET=1 pero eventlet no está instalado: %s. Se usará el modo threading.",
        globals()["_EVENTLET_IMPORT_ERROR"],
    )


def _parse_snapshot_interval(raw: str | None) -> int:
    try:
        value = int(raw) if raw is not None else 60
    except (TypeError, ValueError):
        return 60
    return max(value, 0)


_EXCEL_SNAPSHOT_INTERVAL = _parse_snapshot_interval(
    os.getenv("EXCEL_SNAPSHOT_INTERVAL")
)
_last_excel_snapshot = 0.0


# No caches globales del webapp; usamos import tardío en _notify_dashboard_trade_opened
def _notify_dashboard_trade_opened(
    trade_id: str,
    *,
    trade_details: dict | None = None,
) -> dict | None:
    """Emit dashboard events after a trade is opened."""

    try:
        # Import lazily to avoid capturing stale references during module import.
        from . import webapp  # noqa: WPS433 - delayed import by design
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug(
            "No se pudo cargar el módulo webapp para notificar %s: %s",
            trade_id,
            exc,
        )
        return trade_details or find_trade(trade_id=trade_id)

    details = trade_details or find_trade(trade_id=trade_id)

    try:
        broadcast = getattr(webapp, "broadcast_trades_refresh", None)
        if callable(broadcast):
            broadcast()
        socketio_emit = getattr(webapp, "_emit", None)
        if callable(socketio_emit) and details:
            socketio_emit("trade_updated", {"trade": details})
    except Exception as exc:  # pragma: no cover - notification best-effort
        logger.debug(
            "No se pudo notificar la apertura de %s al dashboard: %s",
            trade_id,
            exc,
        )

    return details


class ModelPerformanceMonitor:
    """Track predictive performance and adjust model weight when degraded."""

    def _empty_metrics(self) -> dict[str, float | int | None]:
        return {
            "count": 0,
            "hit_rate": None,
            "avg_prob": None,
            "drift": None,
            "p_value": None,
        }

    def __init__(
        self,
        window: int,
        min_samples: int,
        min_win_rate: float,
        max_drift: float,
    ) -> None:
        self.samples: deque[tuple[float, int]] = deque(maxlen=window)
        self.min_samples = min_samples
        self.min_win_rate = min_win_rate
        self.max_drift = max_drift
        self._lock = RLock()
        self._degraded = False
        self._latest_metrics: dict[str, float | int | None] = self._empty_metrics()

    def reset(self) -> None:
        with self._lock:
            self.samples.clear()
            self._degraded = False
            self._latest_metrics = self._empty_metrics()

    def record_trade(self, trade: dict | None) -> None:
        if not trade:
            return
        prob = trade.get("model_prob")
        if prob is None:
            return
        outcome = 1 if float(trade.get("profit", 0.0)) > 0 else 0
        with self._lock:
            self.samples.append((float(prob), outcome))
            self._evaluate_locked()

    def _compute_metrics_locked(self) -> dict[str, float | int | None]:
        if not self.samples:
            return self._empty_metrics()
        total = len(self.samples)
        hits = sum(outcome for _, outcome in self.samples)
        hit_rate = hits / total if total else None
        avg_prob = mean(prob for prob, _ in self.samples) if total else None
        drift = (
            abs(avg_prob - hit_rate)
            if avg_prob is not None and hit_rate is not None
            else None
        )
        p_value: float | None = None
        if (
            binomtest is not None
            and hit_rate is not None
            and avg_prob is not None
            and 0.0 < avg_prob < 1.0
        ):
            try:
                base = min(max(avg_prob, 1e-6), 1 - 1e-6)
                p_value = float(binomtest(hits, total, base).pvalue)
            except Exception:  # pragma: no cover - defensive
                p_value = None
        return {
            "count": total,
            "hit_rate": hit_rate,
            "avg_prob": avg_prob,
            "drift": drift,
            "p_value": p_value,
        }

    def _evaluate_locked(self) -> None:
        metrics = self._compute_metrics_locked()
        self._latest_metrics = metrics

        count = int(metrics["count"]) if metrics["count"] is not None else 0
        hit_rate = metrics["hit_rate"]
        avg_prob = metrics["avg_prob"]
        drift = metrics["drift"]

        if (
            count == 0
            or hit_rate is None
            or avg_prob is None
            or drift is None
        ):
            return

        record_model_performance(hit_rate, avg_prob, drift)

        if count < max(self.min_samples, 1):
            return

        current_weight = strategy.get_model_weight()
        if (
            (hit_rate < self.min_win_rate or drift > self.max_drift)
            and current_weight > config.MODEL_WEIGHT_FLOOR
        ):
            new_weight = max(
                config.MODEL_WEIGHT * config.MODEL_WEIGHT_DEGRADATION,
                config.MODEL_WEIGHT_FLOOR,
            )
            strategy.set_model_weight_override(new_weight)
            self._degraded = True
            logger.warning(
                "Model weight degraded to %.2f due to drift (hit_rate=%.2f avg_prob=%.2f drift=%.2f)",
                new_weight,
                hit_rate,
                avg_prob,
                drift,
            )
            self.samples.clear()
            return

        if self._degraded and hit_rate >= self.min_win_rate and drift <= self.max_drift:
            strategy.set_model_weight_override(None)
            self._degraded = False
            logger.info(
                "Model performance recovered (hit_rate=%.2f avg_prob=%.2f); restoring weight %.2f",
                hit_rate,
                avg_prob,
                config.MODEL_WEIGHT,
            )
            self.samples.clear()

    def metrics(self) -> dict[str, float | int | None]:
        with self._lock:
            return dict(self._latest_metrics)


@dataclass
class ShadowPosition:
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    take_profit: float
    stop_loss: float
    open_time: datetime
    max_duration: timedelta
    probabilities: dict[str, float]


SHADOW_COMPARE_MODES = ("heuristic", "hybrid")
_shadow_positions: dict[str, ShadowPosition] = {}
_shadow_lock = RLock()


_model_lock = RLock()
_cached_model = None
_cached_model_mtime: float | None = None
_model_missing_logged = False
_cached_model_is_dummy = False


def _set_cached_model(model) -> None:
    global _cached_model, _cached_model_is_dummy
    with _model_lock:
        _cached_model = model
        _cached_model_is_dummy = bool(getattr(model, "_is_dummy_model", False))


def _model_manifest_timestamp(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def maybe_reload_model(force: bool = False) -> None:
    """Reload the predictive model if the file changed."""

    global _cached_model_mtime, _model_missing_logged, _cached_model_is_dummy
    model_path = config.MODEL_PATH

    if not os.path.exists(model_path):
        if not _model_missing_logged or force:
            logger.warning(
                "Modelo no encontrado en %s; se usará modelo de respaldo",
                model_path,
            )
            _model_missing_logged = True
        if force or not _cached_model_is_dummy or _cached_model is None:
            dummy_model = predictive_model.load_dummy_model()
            _set_cached_model(dummy_model)
            MODEL_MONITOR.reset()
        _cached_model_mtime = None
        return

    mtime = _model_manifest_timestamp(model_path)
    if mtime is None:
        if force or not _model_missing_logged:
            logger.warning("Modelo no disponible en %s", model_path)
            _model_missing_logged = True
        if force or not _cached_model_is_dummy or _cached_model is None:
            dummy_model = predictive_model.load_dummy_model()
            _set_cached_model(dummy_model)
            MODEL_MONITOR.reset()
        _cached_model_mtime = None
        return

    if not force and _cached_model_mtime is not None and mtime <= _cached_model_mtime:
        return

    model = optimizer.load_model(model_path)
    if model is None:
        if not _model_missing_logged or force:
            logger.warning(
                "No se encontró el modelo histórico en %s; las señales se generarán sin ajuste.",
                model_path,
            )
            _model_missing_logged = True
        dummy_model = predictive_model.load_dummy_model()
        _set_cached_model(dummy_model)
        MODEL_MONITOR.reset()
        _cached_model_mtime = None
        return

    _set_cached_model(model)
    _cached_model_mtime = mtime
    _model_missing_logged = False
    strategy.set_model_weight_override(None)
    MODEL_MONITOR.reset()
    logger.info("Modelo predictivo recargado desde %s", model_path)


def current_model():
    with _model_lock:
        return _cached_model


MODEL_MONITOR = ModelPerformanceMonitor(
    config.MODEL_PERFORMANCE_WINDOW,
    config.MODEL_MIN_SAMPLES_FOR_MONITOR,
    config.MODEL_MIN_WIN_RATE,
    config.MODEL_MAX_CALIBRATION_DRIFT,
)


def _snapshot_ai_state() -> None:
    """Persist the latest AI state snapshot to Excel."""

    try:
        model_path = Path(getattr(config, "MODEL_PATH", "models/model.pkl"))
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        try:
            stat = model_path.stat()
        except FileNotFoundError:
            stat = None

        model_info = {
            "model_path": str(model_path),
            "exists": stat is not None,
            "size_bytes": stat.st_size if stat else None,
            "modified_at": (
                datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                if stat
                else None
            ),
            "model_weight": getattr(config, "MODEL_WEIGHT", None),
            "current_weight": strategy.get_model_weight(),
            "mode": getattr(config, "BOT_MODE", None)
            or getattr(config, "TRADING_MODE", None),
        }

        manifest_path = Path(getattr(config, "MODEL_DIR", model_path.parent)) / "manifest.json"
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        training_metrics = {
            "samples": manifest.get("samples"),
            "duration_s": manifest.get("duration_s"),
            "version": manifest.get("version"),
        }
        training_metrics.update(manifest.get("metrics", {}))

        runtime_metrics = MODEL_MONITOR.metrics()
        runtime_metrics.update(
            {
                "auto_trade": getattr(config, "AUTO_TRADE", None),
                "enable_trading": getattr(config, "ENABLE_TRADING", None),
            }
        )

        exporter.write_ai_status(
            model_info=model_info,
            training_metrics=training_metrics,
            runtime_metrics=runtime_metrics,
        )
    except Exception as exc:  # pragma: no cover - defensive snapshot
        logger.debug("Failed to export AI snapshot: %s", exc, exc_info=exc)


def _snapshot_ops_state() -> None:
    """Persist operational telemetry to Excel."""

    try:
        positions = all_open_trades()
        risk_limits = {
            "max_daily_loss_usdt": getattr(config, "MAX_DAILY_LOSS_USDT", None),
            "min_position_size_usdt": getattr(config, "MIN_POSITION_SIZE_USDT", None),
            "max_open_trades": getattr(config, "MAX_OPEN_TRADES", None),
        }
        ws_client = globals().get("WS_CLIENT")
        ws_status = {
            "connected": getattr(ws_client, "is_connected", None)
            if ws_client is not None
            else None,
            "last_heartbeat": getattr(ws_client, "last_ping", None)
            if ws_client is not None
            else None,
        }
        exporter.write_ops_snapshot(
            positions=positions,
            risk_limits=risk_limits,
            ws_status=ws_status,
        )
    except Exception as exc:  # pragma: no cover - defensive snapshot
        logger.debug("Failed to export OPS snapshot: %s", exc, exc_info=exc)


def _reset_shadow_positions() -> None:
    with _shadow_lock:
        _shadow_positions.clear()


def _record_shadow_payloads(position: ShadowPosition, signal: dict) -> None:
    base_payload = dict(signal)
    for mode in SHADOW_COMPARE_MODES:
        payload = dict(base_payload)
        if mode == "heuristic":
            prob = position.probabilities.get(mode)
            if "orig_prob" in payload:
                payload["prob_success"] = payload["orig_prob"]
            elif prob is not None:
                payload["prob_success"] = prob
        else:
            prob = position.probabilities.get(mode)
            if prob is not None:
                payload["prob_success"] = prob
        payload["trade_id"] = f"{position.trade_id}:{mode}"
        payload["mode"] = mode
        shadow.record_shadow_signal(position.symbol, mode, payload)


def _register_shadow_trade(signal: dict) -> None:
    trade_uuid = str(uuid.uuid4())
    entry_price = float(signal.get("entry_price", 0.0))
    quantity = float(signal.get("quantity", 0.0))
    take_profit = float(signal.get("take_profit", entry_price))
    stop_loss = float(signal.get("stop_loss", entry_price))
    try:
        max_minutes = int(
            signal.get("max_duration_minutes", config.MAX_TRADE_DURATION_MINUTES)
        )
    except (TypeError, ValueError):
        max_minutes = config.MAX_TRADE_DURATION_MINUTES
    if max_minutes <= 0:
        max_minutes = config.MAX_TRADE_DURATION_MINUTES

    probabilities = {
        "hybrid": float(signal.get("prob_success", 0.0)),
        "heuristic": float(
            signal.get("orig_prob", signal.get("prob_success", 0.0))
        ),
    }

    position = ShadowPosition(
        trade_id=trade_uuid,
        symbol=str(signal.get("symbol", "")),
        side=str(signal.get("side", "BUY")),
        entry_price=entry_price,
        quantity=quantity,
        take_profit=take_profit,
        stop_loss=stop_loss,
        open_time=datetime.now(timezone.utc),
        max_duration=timedelta(minutes=max_minutes),
        probabilities=probabilities,
    )

    with _shadow_lock:
        _shadow_positions[trade_uuid] = position

    _record_shadow_payloads(position, signal)


def _assess_shadow_close(
    position: ShadowPosition, price: float, now: datetime
) -> tuple[str | None, float, float]:
    side = position.side.upper()
    hit_tp = price >= position.take_profit if side == "BUY" else price <= position.take_profit
    hit_sl = price <= position.stop_loss if side == "BUY" else price >= position.stop_loss

    reason: str | None = None
    if hit_tp:
        reason = "TP"
    elif hit_sl:
        reason = "SL"
    elif now - position.open_time >= position.max_duration:
        reason = "MAX_DURATION"

    if reason is None:
        return None, price, 0.0

    if side == "BUY":
        profit = (price - position.entry_price) * position.quantity
    else:
        profit = (position.entry_price - price) * position.quantity

    return reason, price, profit


def _finalize_shadow_trade(
    position: ShadowPosition,
    reason: str,
    exit_price: float,
    profit: float,
    closed_at: datetime,
) -> None:
    payload = {
        "symbol": position.symbol,
        "side": position.side,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "quantity": position.quantity,
        "profit": profit,
        "close_reason": reason,
        "closed_at": closed_at.isoformat().replace("+00:00", "Z"),
    }
    for mode in SHADOW_COMPARE_MODES:
        result = dict(payload)
        prob = position.probabilities.get(mode)
        if prob is not None:
            result["prob_success"] = prob
        result["mode"] = mode
        shadow.finalize_shadow_trade(f"{position.trade_id}:{mode}", result)


def _process_shadow_positions() -> float:
    with _shadow_lock:
        snapshot = list(_shadow_positions.items())

    realized = 0.0
    completed: list[str] = []
    for trade_id, position in snapshot:
        price_raw = data.get_current_price_ticker(position.symbol)
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            continue

        now = datetime.now(timezone.utc)
        reason, exit_price, profit = _assess_shadow_close(position, price, now)
        if reason is None:
            continue

        _finalize_shadow_trade(position, reason, exit_price, profit, now)
        logger.info(
            "Shadow trade %s %s closed (%s) profit %.4f",
            position.symbol,
            position.side,
            reason,
            profit,
        )
        realized += profit
        completed.append(trade_id)

    if completed:
        with _shadow_lock:
            for trade_id in completed:
                _shadow_positions.pop(trade_id, None)

    return realized


def parse_args():
    """Parse CLI options controlling runtime mode selection."""

    parser = argparse.ArgumentParser(description="Trading bot runner")
    parser.add_argument(
        "--mode",
        choices=sorted(bot_mode.MODES.keys()),
        help="Selecciona modo de ejecución (override de ENV/menú)",
    )
    parser.add_argument(
        "--no-interactive-menu",
        action="store_true",
        help="Desactiva el menú interactivo incluso si hay TTY",
    )
    parser.add_argument(
        "--backtest-config",
        type=str,
        help="Ruta al YAML de configuración del backtest (modo backtest)",
    )
    parser.add_argument(
        "--backtest-data",
        type=str,
        help="Ruta al CSV con datos para el backtest (modo backtest)",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Abre el dashboard como aplicación de escritorio (PyWebview)",
    )
    return parser.parse_args()


def _run_backtest_startup(args) -> None:
    """Execute the backtest runner when the selected mode requests it."""

    from . import backtest

    cfg_path_str = (
        args.backtest_config
        or config.BACKTEST_CONFIG_PATH
        or os.getenv("BACKTEST_CONFIG_PATH")
        or "backtest.yml"
    )
    data_path_str = (
        args.backtest_data
        or config.BACKTEST_DATA_PATH
        or os.getenv("BACKTEST_DATA_PATH")
        or "backtest.csv"
    )
    cfg_path = Path(cfg_path_str)
    data_path = Path(data_path_str)

    if not cfg_path.exists():
        logger.error("Backtest config not found: %s", cfg_path)
        return
    if not data_path.exists():
        logger.error("Backtest dataset not found: %s", data_path)
        return

    try:
        cfg = backtest._load_config(cfg_path)
        dataset = backtest._load_dataset(data_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unable to load backtest resources: %s", exc)
        return

    kpis = backtest.run_backtest(cfg, dataset)
    logger.info("Backtest KPIs: %s", kpis)


def open_new_trade(signal: dict):
    """Open a position and track its state via ``trade_manager``."""
    symbol = signal["symbol"]
    raw = normalize_symbol(symbol).replace("_", "")

    if config.SHADOW_MODE:
        _register_shadow_trade(signal)
        return None

    # Register trade in pending state first
    trade = add_trade(signal)
    try:
        if not config.DRY_RUN:
            execution.setup_leverage(execution.exchange, raw, signal["leverage"])
        else:
            logger.info("Skip leverage setup for %s (dry-run)", symbol)
        order = execution.open_position(
            symbol,
            signal["side"],
            signal["quantity"],
            signal["entry_price"],
            order_type="limit",
        )
        if not isinstance(order, dict):
            logger.warning("Order response unexpected for %s: %s", symbol, order)
            set_trade_state(trade["trade_id"], TradeState.FAILED)
            return None

        order_id = order.get("id")
        avg_price = float(order.get("average") or signal["entry_price"])
        update_trade(
            trade["trade_id"],
            order_id=order_id,
            entry_price=avg_price,
            status="active",
            created_ts=time.time(),
        )

        if config.DRY_RUN and os.getenv("TEST_MODE") != "1":
            opened_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            update_trade(trade["trade_id"], open_time=opened_at)
            set_trade_state(trade["trade_id"], TradeState.OPEN)
            save_trades()
            details = find_trade(trade_id=trade["trade_id"])
            # Notificar SIEMPRE antes de salir (un solo return)
            return _notify_dashboard_trade_opened(trade["trade_id"], trade_details=details) or details

        status = execution.fetch_order_status(order_id, symbol) if order_id else "new"
        if status == "filled":
            set_trade_state(trade["trade_id"], TradeState.OPEN)
        elif status == "partial":
            # promote to OPEN first, then PARTIALLY_FILLED
            set_trade_state(trade["trade_id"], TradeState.OPEN)
            set_trade_state(trade["trade_id"], TradeState.PARTIALLY_FILLED)
        # else: remain in PENDING

        if status in {"filled", "partial"} and order_id:
            details = execution.get_order_fill_details(order_id, symbol)
            if details:
                updates: dict[str, float] = {}
                filled_qty = details.get("filled")
                if filled_qty is not None and filled_qty > 0:
                    updates["quantity"] = filled_qty
                remaining_qty = details.get("remaining")
                if remaining_qty is not None:
                    updates["remaining_quantity"] = max(remaining_qty, 0.0)
                avg_exec = details.get("average")
                if avg_exec:
                    updates.setdefault("entry_price", avg_exec)
                if updates:
                    update_trade(trade["trade_id"], **updates)

        save_trades()
        details = find_trade(trade_id=trade["trade_id"])
        # Notificar SIEMPRE antes de salir (un solo return)
        return _notify_dashboard_trade_opened(trade["trade_id"], trade_details=details) or details

    except permissions.PermissionError as exc:
        logger.error("Permission denied opening %s: %s", symbol, exc)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    except execution.OrderSubmitError:
        logger.error("Order submission failed for %s", symbol)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    except Exception as exc:
        logger.error("Error processing %s: %s", symbol, exc)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    return None


def close_existing_trade(
    trade: dict, reason: str
) -> tuple[dict | None, float | None, float | None]:
    """Close a trade enforcing state transitions and return results."""

    trade_id = trade.get("trade_id")
    symbol = trade.get("symbol")
    qty = float(trade.get("quantity", 0.0))
    close_side = "close_short" if trade.get("side") == "SELL" else "close_long"

    set_trade_state(trade_id, TradeState.CLOSING)
    try:
        order = execution.close_position(symbol, close_side, qty, order_type="market")
    except execution.OrderSubmitError:
        logger.error("Order submission failed when closing %s", symbol)
        set_trade_state(trade_id, TradeState.FAILED)
        save_trades()
        return None, None, None


    order_id = order.get("id")
    exec_price = float(order.get("average") or order.get("price") or trade.get("entry_price", 0.0))

    if order_id:
        attempts = max(1, config.API_RETRY_ATTEMPTS)
        for attempt in range(attempts):
            status = execution.fetch_order_status(order_id, symbol)
            if status == "filled":
                break
            if status == "partial":
                details = execution.get_order_fill_details(order_id, symbol)
                remaining_qty = 0.0
                if details:
                    remaining_qty = max(details.get("remaining", 0.0) or 0.0, 0.0)
                    if details.get("average"):
                        exec_price = float(details["average"])
                if remaining_qty <= config.CLOSE_REMAINING_TOLERANCE:
                    break
                logger.warning(
                    "Partial close for %s (remaining %.6f); reattempting", symbol, remaining_qty
                )
                try:
                    order = execution.close_position(
                        symbol, close_side, remaining_qty, order_type="market"
                    )
                    order_id = order.get("id", order_id)
                    if order.get("average"):
                        exec_price = float(order.get("average"))
                except execution.OrderSubmitError:
                    logger.error("Failed to close remaining %.6f for %s", remaining_qty, symbol)
                    break
            else:
                break
            time.sleep(0.5 * (attempt + 1))

    remaining = execution.fetch_position_size(symbol)
    if remaining > config.CLOSE_REMAINING_TOLERANCE:
        logger.warning(
            "Remanente %.6f detectado tras cierre de %s; trade marcado como parcial",
            remaining,
            symbol,
        )
        update_trade(trade_id, quantity=remaining)
        set_trade_state(trade_id, TradeState.PARTIALLY_FILLED)
        save_trades()
        return None, exec_price, None

    entry_price = float(trade.get("entry_price", exec_price))
    if trade.get("side") == "BUY":
        realized = (exec_price - entry_price) * qty
    else:
        realized = (entry_price - exec_price) * qty

    closed = close_trade(
        trade_id=trade_id,
        reason=reason,
        exit_price=exec_price,
        profit=realized,
    )
    MODEL_MONITOR.record_trade(closed)
    if closed:
        try:
            trade_copy = dict(closed)
            if exec_price is not None and trade_copy.get("exit_price") is None:
                trade_copy["exit_price"] = exec_price
            if realized is not None:
                trade_copy.setdefault("pnl", realized)
                trade_copy.setdefault("profit", realized)
            path = exporter.append_trade_closed(trade_copy)
            logger.info("Trade exportado a Excel: %s", path)
        except Exception as exc:  # pragma: no cover - defensive export
            logger.warning("No se pudo exportar trade a Excel: %s", exc)
    auto_trainer.record_completed_trade(closed)
    save_trades()
    return closed, exec_price, realized


def run_one_iteration_open(model=None):
    """Execute a single iteration of the opening logic."""
    if model is None:
        model = current_model()
    execution.cleanup_old_orders()
    if count_open_trades() >= config.MAX_OPEN_TRADES:
        return
    symbols = data.get_common_top_symbols(execution.exchange, 15)
    candidates = []
    seen = set()
    for symbol in symbols:
        if find_trade(symbol=symbol) or trade_manager.in_cooldown(symbol):
            continue
        norm = trade_manager.normalize_symbol(symbol)
        if norm in seen:
            continue
        seen.add(norm)
        raw = symbol.replace("_", "")
        if raw in config.BLACKLIST_SYMBOLS or raw in config.UNSUPPORTED_SYMBOLS:
            continue
        sig = strategy.decidir_entrada(symbol, modelo_historico=model)
        if not sig:
            continue
        if sig.get("risk_reward", 0) < config.MIN_RISK_REWARD:
            logger.debug(
                "Skip %s: RR=%.2f < %.2f", symbol, sig.get("risk_reward", 0), config.MIN_RISK_REWARD
            )
            continue
        if sig.get("quantity", 0) < config.MIN_POSITION_SIZE:
            logger.debug(
                "Skip %s: qty=%.8f < min=%.8f",
                symbol,
                sig.get("quantity", 0),
                config.MIN_POSITION_SIZE,
            )
            continue
        candidates.append(sig)

    candidates.sort(key=lambda s: s.get("prob_success", 0), reverse=True)
    for sig in candidates:
        if count_open_trades() >= config.MAX_OPEN_TRADES:
            break
        try:
            open_new_trade(sig)
        except Exception as exc:
            logger.error("Error processing %s: %s", sig.get("symbol"), exc)

def run(*, use_desktop: bool = False, install_signal_handlers: bool = True) -> None:
    global _last_excel_snapshot
    load_trades()  # Restaurar operaciones guardadas
    permissions.audit_environment(execution.exchange)

    if execution.exchange is None:
        logger.error("Conexión al exchange no inicializada. Abortando trading.")
        config.ENABLE_TRADING = False
        config.AUTO_TRADE = False
        config.MAINTENANCE = True
        maybe_alert(True, "❌ No se pudo conectar al exchange. Trading pausado.")
        return

    # Sincronizar con posiciones reales en el exchange
    positions = execution.fetch_positions()
    active_symbols = set()
    for pos in positions:
        symbol = normalize_symbol(pos.get("symbol", ""))
        qty = float(pos.get("contracts", 0))
        if qty == 0:
            continue
        active_symbols.add(symbol)
        trade = find_trade(symbol=symbol)
        if trade:
            trade.update(
                quantity=qty,
                entry_price=float(pos.get("entryPrice", trade.get("entry_price", 0))),
                leverage=int(pos.get("leverage", trade.get("leverage", 1))),
                status="active",
            )
        else:
            side_raw = str(pos.get("side", "")).lower()
            is_long = side_raw == "long"
            try:
                entry_price = float(pos.get("entryPrice") or 0.0)
            except (TypeError, ValueError):
                entry_price = 0.0
            try:
                stop_loss_exchange = float(pos.get("stopLossPrice") or 0.0)
            except (TypeError, ValueError):
                stop_loss_exchange = 0.0
            try:
                take_profit_exchange = float(pos.get("takeProfitPrice") or 0.0)
            except (TypeError, ValueError):
                take_profit_exchange = 0.0

            stop_loss_fallback = entry_price * (0.98 if is_long else 1.02)
            take_profit_fallback = entry_price * (1.02 if is_long else 0.98)

            trade = {
                "symbol": symbol,
                "side": "BUY" if is_long else "SELL",
                "quantity": qty,
                "entry_price": entry_price,
                "stop_loss": stop_loss_exchange or stop_loss_fallback,
                "take_profit": take_profit_exchange or take_profit_fallback,
                "leverage": int(pos.get("leverage", 1)),
                "status": "active",
            }
            add_trade(trade)

    # Eliminar operaciones locales que no existan en el exchange
    for tr in list(all_open_trades()):
        if tr["symbol"] not in active_symbols:
            close_trade(trade_id=tr.get("trade_id"), reason="sync")

    # Cancelar órdenes abiertas pendientes no registradas
    for order in execution.fetch_open_orders():
        sym = normalize_symbol(order.get("symbol", ""))
        if sym not in active_symbols:
            execution.cancel_order(order.get("id"), sym)

    # Remove any stale pending orders
    execution.cleanup_old_orders()

    # Persist state after initial synchronization
    save_trades()

    # Launch the dashboard using trade_manager as the single source of trades
    # (no operations list is passed).
    if not use_desktop:
        Thread(
            target=webapp.start_dashboard,
            args=(config.WEBAPP_HOST, config.WEBAPP_PORT),
            daemon=True,
        ).start()

    # Expose Prometheus metrics and start system monitor
    Thread(target=start_metrics_server, args=(config.METRICS_PORT,), daemon=True).start()
    Thread(target=monitor_system, daemon=True).start()

    strategy.start_liquidity()

    stop_event = shutdown.get_stop_event()
    stop_event.clear()
    trainer = auto_trainer.start_auto_trainer(stop_event)

    if install_signal_handlers:
        shutdown.install_signal_handlers()
    shutdown.register_callback(save_trades)
    shutdown.register_callback(execution.cancel_all_orders)
    if trainer is not None:
        def _stop_trainer() -> None:
            stop_event.set()
            trainer.join(timeout=10)

        shutdown.register_callback(_stop_trainer)

    maybe_reload_model(force=True)
    daily_profit = 0.0
    trading_active = bool(config.ENABLE_TRADING or config.SHADOW_MODE)

    current_day = datetime.now(timezone.utc).date()
    loss_limit = -abs(config.MAX_DAILY_LOSS_USDT)
    standby_notified = False
    shutdown_handled = False
    drift_warning_sent = False
    drift_notified = False

    # Initial metric update
    update_trade_metrics(count_open_trades(), len(trade_manager.all_closed_trades()))

    logger.info("Starting trading loop...")
    logger.info(
        "Autonomía: AUTO_TRADE=%s | KillSwitch drift=%s (p_crit=%.3f, hit_warn=%.2f, hit_crit=%.2f)",
        config.AUTO_TRADE,
        config.KILL_SWITCH_ON_DRIFT,
        config.DRIFT_PVALUE_CRIT,
        config.HIT_RATE_ROLLING_WARN,
        config.HIT_RATE_ROLLING_CRIT,
    )
    logger.info(
        "Risk limits: max_daily_loss=%.2f USDT | min_notional=%.2f | max_positions=%d",
        config.MAX_DAILY_LOSS_USDT,
        config.MIN_POSITION_SIZE_USDT,
        config.MAX_OPEN_TRADES,
    )

    _snapshot_ai_state()
    _snapshot_ops_state()
    _last_excel_snapshot = time.time()
    last_reconcile = 0.0

    while True:
        try:

            execution.cleanup_old_orders()

            now_ts = time.time()
            if now_ts - last_reconcile >= config.RECONCILE_INTERVAL_S:
                reconcile_pending_trades()
                last_reconcile = now_ts

            maybe_reload_model()
            model = current_model()

            if _EXCEL_SNAPSHOT_INTERVAL > 0:
                ts_now = time.time()
                if ts_now - _last_excel_snapshot >= _EXCEL_SNAPSHOT_INTERVAL:
                    _snapshot_ai_state()
                    _snapshot_ops_state()
                    _last_excel_snapshot = ts_now

            if shutdown.shutdown_requested() and not shutdown_handled:
                logger.info("Shutdown requested; stopping trading loop")
                trading_active = False
                shutdown_handled = True
                shutdown.execute_callbacks()
                break

            now = datetime.now(timezone.utc)
            if now.date() != current_day:
                current_day = now.date()
                daily_profit = 0.0
                trading_active = bool(config.ENABLE_TRADING or config.SHADOW_MODE)
                loss_limit = -abs(config.MAX_DAILY_LOSS_USDT)

                standby_notified = False
                logger.info("New trading day detected; counters reset")

            # Detener nuevas entradas cuando la pérdida diaria alcanza el límite
            # Se acepta que DAILY_RISK_LIMIT pueda definirse como valor negativo
            # (por ejemplo -50.0) o positivo (50.0). Siempre se compara contra
            # el valor negativo correspondiente.
            if daily_profit <= loss_limit and trading_active:
                trading_active = False
                standby_notified = False
                logger.error("Daily loss limit reached %.2f", daily_profit)
                maybe_alert(True, f"Daily loss limit reached {daily_profit:.2f} USDT")

            drift_blocked = False
            metrics_snapshot = MODEL_MONITOR.metrics()
            auto_trainer.observe_live_metrics(metrics_snapshot)
            drift_hit_rate = metrics_snapshot.get("hit_rate")
            drift_p_value = metrics_snapshot.get("p_value")
            drift_count = int(metrics_snapshot.get("count") or 0)

            if (
                config.KILL_SWITCH_ON_DRIFT
                and drift_count >= config.MODEL_MIN_SAMPLES_FOR_MONITOR
            ):
                if (
                    drift_p_value is not None
                    and drift_p_value <= config.DRIFT_PVALUE_CRIT
                ):
                    drift_blocked = True
                if (
                    drift_hit_rate is not None
                    and drift_hit_rate <= config.HIT_RATE_ROLLING_CRIT
                ):
                    drift_blocked = True
                if (
                    not drift_blocked
                    and drift_hit_rate is not None
                    and drift_hit_rate <= config.HIT_RATE_ROLLING_WARN
                ):
                    if not drift_warning_sent:
                        warning_msg = (
                            "Model hit rate warning: "
                            f"{drift_hit_rate:.2%} <= {config.HIT_RATE_ROLLING_WARN:.2%}"
                        )
                        logger.warning(warning_msg)
                        maybe_alert(True, warning_msg, cooldown=900)
                        drift_warning_sent = True
                elif (
                    drift_hit_rate is not None
                    and drift_hit_rate > config.HIT_RATE_ROLLING_WARN
                ):
                    drift_warning_sent = False
            else:
                if not config.KILL_SWITCH_ON_DRIFT:
                    drift_notified = False
                drift_warning_sent = False

            if drift_blocked:
                if not drift_notified:
                    hit_rate_str = (
                        f"{drift_hit_rate:.2%}" if drift_hit_rate is not None else "n/a"
                    )
                    p_value_str = (
                        f"{drift_p_value:.4f}" if drift_p_value is not None else "n/a"
                    )
                    kill_msg = (
                        "Kill switch engaged due to drift "
                        f"(hit_rate={hit_rate_str}, p_value={p_value_str})"
                    )
                    logger.error(kill_msg)
                    maybe_alert(True, kill_msg, cooldown=900)
                    drift_notified = True
            elif drift_notified and config.KILL_SWITCH_ON_DRIFT:
                recovery_msg = "Drift kill switch cleared; autonomy restored"
                logger.info(recovery_msg)
                maybe_alert(True, recovery_msg, cooldown=900)
                drift_notified = False

            # ABRIR NUEVAS OPERACIONES (solo si hay hueco y permitido)
            allow_candidates = (
                trading_active
                and not config.MAINTENANCE
                and (config.ENABLE_TRADING or config.SHADOW_MODE)
                and config.AUTO_TRADE
                and not drift_blocked
                and count_open_trades() < config.MAX_OPEN_TRADES
            )

            if allow_candidates:
                if not config.SHADOW_MODE and not permissions.can_open_trade(execution.exchange):
                    logger.debug("Skipping entries: live trading permissions not granted")
                else:
                    if config.TEST_MODE and config.TEST_SYMBOLS:
                        symbols = [
                            s.replace("/", "_").replace("-", "_")
                            for s in config.TEST_SYMBOLS
                        ]
                    elif config.BOT_MODE == "shadow" and config.SYMBOLS:
                        symbols = [
                            s.replace("/", "_").replace("-", "_")
                            for s in config.SYMBOLS
                        ]
                    else:
                        symbols = data.get_common_top_symbols(execution.exchange, 15)
                    candidates = []
                    seen = set()
                    for symbol in symbols:
                        norm_symbol = trade_manager.normalize_symbol(symbol)
                        if norm_symbol in seen:
                            continue
                        seen.add(norm_symbol)
                        # enforce per-symbol trade limit
                        if count_trades_for_symbol(symbol) >= config.MAX_TRADES_PER_SYMBOL:
                            continue
                        raw = normalize_symbol(symbol).replace("_", "")
                        bl = {normalize_symbol(s).replace("_", "") for s in config.BLACKLIST_SYMBOLS}
                        unsup = {normalize_symbol(s).replace("_", "") for s in config.UNSUPPORTED_SYMBOLS}
                        if raw in bl or raw in unsup:
                            continue
                        if trade_manager.in_cooldown(symbol):
                            logger.info("Cooldown activo para %s; se omite nueva entrada", symbol)
                            continue
                        sig = strategy.decidir_entrada(symbol, modelo_historico=model)
                        if not sig:
                            continue
                        if sig.get("risk_reward", 0) < config.MIN_RISK_REWARD:
                            logger.debug(
                                "Skip %s: RR=%.2f < %.2f",
                                symbol,
                                sig.get("risk_reward", 0),
                                config.MIN_RISK_REWARD,
                            )
                            continue
                        if sig.get("quantity", 0) < config.MIN_POSITION_SIZE:
                            logger.debug(
                                "Skip %s: qty=%.8f < min=%.8f",
                                symbol,
                                sig.get("quantity", 0),
                                config.MIN_POSITION_SIZE,
                            )
                            continue
                        notional = float(sig.get("quantity", 0.0)) * float(
                            sig.get("entry_price", 0.0)
                        )
                        if notional < config.MIN_POSITION_SIZE_USDT:
                            logger.debug(
                                "Skip %s: notional %.2f < min_usdt=%.2f",
                                symbol,
                                notional,
                                config.MIN_POSITION_SIZE_USDT,
                            )
                            continue
                        candidates.append(sig)

                    candidates.sort(key=lambda s: s.get("prob_success", 0), reverse=True)
                    for sig in candidates:
                        if count_open_trades() >= config.MAX_OPEN_TRADES:
                            break
                        symbol = sig["symbol"]
                        raw = normalize_symbol(symbol).replace("_", "")
                        try:
                            trade = open_new_trade(sig)
                            if not trade:
                                continue
                            notify.send_telegram(
                                f"Opened {symbol} {trade['side']} @ {trade['entry_price']}"
                            )
                            notify.send_discord(
                                f"Opened {symbol} {trade['side']} @ {trade['entry_price']}"
                            )
                            logger.info("Opened trade: %s", trade)
                        except permissions.PermissionError as exc:
                            logger.error("Permission denied for %s: %s", symbol, exc)
                            break
                        except Exception as exc:
                            logger.error("Error processing %s: %s", symbol, exc)
            else:
                if not config.AUTO_TRADE:
                    logger.debug("Skipping entries: AUTO_TRADE disabled")
                elif drift_blocked:
                    logger.warning("Skipping entries: drift kill switch active")
                elif (
                    trading_active
                    and config.ENABLE_TRADING
                    and not config.SHADOW_MODE
                    and not permissions.can_open_trade(execution.exchange)
                ):
                    logger.debug(
                        "Skipping entries: live trading permissions not granted"
                    )

            # MONITOREAR OPERACIONES ABIERTAS
            if config.SHADOW_MODE:
                realized = _process_shadow_positions()
                daily_profit += realized
            else:
                for op in list(all_open_trades()):
                    price = data.get_current_price_ticker(op["symbol"])
                    if not price:
                        continue
                    try:
                        entry_price = float(op.get("entry_price") or 0.0)
                    except (TypeError, ValueError):
                        entry_price = 0.0
                    try:
                        current_stop = float(op.get("stop_loss") or 0.0)
                    except (TypeError, ValueError):
                        current_stop = 0.0
                    try:
                        quantity = float(op.get("quantity") or 0.0)
                    except (TypeError, ValueError):
                        quantity = 0.0
                    try:
                        take_profit_value = float(op.get("take_profit") or 0.0)
                    except (TypeError, ValueError):
                        take_profit_value = 0.0
                    side = str(op.get("side", "BUY")).upper()
                    if (
                        config.TRAILING_STOP_ENABLED
                        and entry_price > 0
                        and price > 0
                        and op.get("trade_id")
                    ):
                        gain_pct = 0.0
                        candidate_stop = None
                        applied_stop = None

                        if side == "BUY":
                            gain_pct = (price / entry_price) - 1.0
                            if gain_pct >= config.TRAILING_STOP_TRIGGER:
                                candidate_stop = max(
                                    entry_price,
                                    price * (1 - config.TRAILING_STOP_DISTANCE),
                                )
                                if candidate_stop <= 0:
                                    candidate_stop = None
                        else:
                            if price > 0:
                                gain_pct = (entry_price - price) / entry_price
                            if gain_pct >= config.TRAILING_STOP_TRIGGER:
                                candidate_stop = min(
                                    entry_price,
                                    price * (1 + config.TRAILING_STOP_DISTANCE),
                                )
                                if candidate_stop <= 0:
                                    candidate_stop = None
                        if candidate_stop is not None:
                            if side == "BUY":
                                if candidate_stop > current_stop * (1 + 1e-6):
                                    if trade_manager.update_trade(
                                        op["trade_id"], stop_loss=candidate_stop
                                    ):
                                        applied_stop = candidate_stop

                            else:
                                # For SELL trades the stop trails downwards while staying above price
                                if current_stop == 0.0 or candidate_stop < current_stop * (1 - 1e-6):
                                    if trade_manager.update_trade(
                                        op["trade_id"], stop_loss=candidate_stop
                                    ):

                                        applied_stop = candidate_stop
                        if applied_stop is not None:
                            current_stop = applied_stop
                            op["stop_loss"] = applied_stop
                            logger.info(
                                "Stop-loss de %s movido a %.4f (trailing)",
                                    op["symbol"],
                                    applied_stop,
                                )
                    try:
                        stop_loss_value = float(op.get("stop_loss") or 0.0)
                    except (TypeError, ValueError):
                        stop_loss_value = 0.0
                    try:
                        op_state = TradeState(op.get("state"))
                    except ValueError:
                        op_state = TradeState.PENDING
                    if op_state not in {TradeState.OPEN, TradeState.PARTIALLY_FILLED}:
                        continue
                    if side == "BUY":
                        profit = (price - entry_price) * quantity
                        close = (
                            (stop_loss_value and price <= stop_loss_value)
                            or (take_profit_value and price >= take_profit_value)
                        )
                    else:
                        profit = (entry_price - price) * quantity
                        close = (
                            (stop_loss_value and price >= stop_loss_value)
                            or (take_profit_value and price <= take_profit_value)
                        )

                    reason = None
                    if close:
                        reason = "TP" if profit >= 0 else "SL"
                    if reason is None and trade_manager.exceeded_max_duration(op):
                        close = True
                        reason = "MAX_DURATION"

                    if close:
                        if side == "BUY":
                            target_hit = take_profit_value > 0 and price >= take_profit_value
                        else:
                            target_hit = take_profit_value > 0 and price <= take_profit_value
                        expected = take_profit_value if target_hit else stop_loss_value
                        close_label = reason or ("TP" if profit >= 0 else "SL")
                        closed, exec_price, realized = close_existing_trade(
                            op,
                            close_label,
                        )
                        if not closed or exec_price is None or realized is None:
                            continue
                        slippage = exec_price - expected
                        logger.info(
                            "Closed %s at %.4f (target %.4f) slippage %.4f",
                            op["symbol"], exec_price, expected, slippage
                        )
                        if abs(slippage) > config.MAX_SLIPPAGE:
                            logger.warning("High slippage detected on %s: %.4f", op["symbol"], slippage)
                        daily_profit += realized

                        outcome = close_label
                        note = (
                            f"Closed {op['symbol']} {outcome} "
                            f"PnL {realized:.2f} Slippage {slippage:.4f}"
                        )
                        notify.send_telegram(note)
                        notify.send_discord(note)

            save_trades()  # Guarda el estado periódicamente
            update_trade_metrics(count_open_trades(), len(trade_manager.all_closed_trades()))

            if not trading_active and count_open_trades() == 0 and not standby_notified:
                logger.info("All positions closed after reaching daily limit")
                save_trades()
                standby_notified = True
            time.sleep(max(1, int(config.LOOP_INTERVAL)))
        except KeyboardInterrupt:

            # On manual interrupt, cancel any pending orders and persist trades
            for order in execution.fetch_open_orders():
                sym = order.get("symbol", "").replace("/", "_").replace(":USDT", "")
                execution.cancel_order(order.get("id"), sym)
            save_trades()

            break
        except Exception as exc:
            logger.error("Loop error: %s", exc)
            time.sleep(10)

    logger.info("Trading loop exited")

def main() -> None:
    """Entry point that selects runtime mode and starts the bot."""

    args = parse_args()
    env_mode = config.BOT_MODE
    if args.mode:
        chosen = bot_mode.resolve_mode(args.mode, None)
    elif not args.no_interactive_menu and sys.stdin.isatty() and not env_mode:
        chosen = bot_mode.interactive_pick()
    else:
        chosen = bot_mode.resolve_mode(None, env_mode)

    bot_mode.apply_mode_to_config(chosen, config)

    desktop_mode = bool(getattr(args, "desktop", False))
    desktop_module = None
    if desktop_mode:
        try:
            from . import desktop as desktop_module  # noqa: WPS433 - optional import
        except Exception as exc:
            logger.exception("No se pudo iniciar el modo escritorio: %s", exc)
            desktop_mode = False

    if config.RUN_BACKTEST_ON_START:
        _run_backtest_startup(args)
        return

    if desktop_mode and desktop_module:
        shutdown.install_signal_handlers()
        trading_thread = Thread(
            target=run,
            kwargs={"use_desktop": True, "install_signal_handlers": False},
            daemon=True,
        )
        trading_thread.start()
        try:
            desktop_module.launch_desktop()
        except Exception:
            logger.exception("Error al lanzar la interfaz de escritorio")
        finally:
            shutdown.request_shutdown()
            trading_thread.join(timeout=30)
            if trading_thread.is_alive():
                logger.warning(
                    "El hilo de trading sigue activo tras cerrar la ventana de escritorio"
                )
        return

    run()


if __name__ == "__main__":
    main()
