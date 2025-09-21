"""Walk-forward backtest utilities for the trading bot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import config, strategy


@dataclass
class BacktestConfig:
    mode: Literal["walk_forward", "rolling"] = "walk_forward"
    train_start: datetime | str | None = None
    test_start: datetime | str | None = None
    test_end: datetime | str | None = None
    rolling_train_days: int = 30
    rolling_test_days: int = 7
    fees: float = config.FEE_EST
    slippage_bps: float = 0.0
    latency_ms: int = 0
    model_weight: float = config.MODEL_WEIGHT
    min_prob_success: float = config.MIN_PROB_SUCCESS
    rr_floor: float = config.MIN_RISK_REWARD


@dataclass
class TradeResult:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    probability: float
    risk_reward: float
    outcome: float
    profit: float


def _parse_timestamp(value: datetime | str | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(fallback.tz)
    return ts.tz_convert(fallback.tz) if fallback.tz is not None else ts


def _iter_splits(frame: pd.DataFrame, cfg: BacktestConfig):
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.assign(timestamp=timestamps).sort_values("timestamp")
    overall_start = timestamps.min()
    test_start = _parse_timestamp(cfg.test_start, overall_start)
    test_end = _parse_timestamp(cfg.test_end, timestamps.max())
    pointer = test_start
    while pointer < test_end:
        if cfg.mode == "rolling":
            train_start = pointer - timedelta(days=cfg.rolling_train_days)
        else:
            train_start = _parse_timestamp(cfg.train_start, overall_start)
        train_mask = (timestamps >= train_start) & (timestamps < pointer)
        test_end_split = min(pointer + timedelta(days=cfg.rolling_test_days), test_end)
        test_mask = (timestamps >= pointer) & (timestamps < test_end_split)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            pointer = test_end_split
            continue
        yield frame.loc[train_mask], frame.loc[test_mask]
        pointer = test_end_split


def blend_probability(orig: float, model: float | None, weight: float) -> float:
    return strategy.blend_probabilities(orig, model, weight)


def simulate_decision(row: pd.Series, cfg: BacktestConfig) -> dict | None:
    risk_reward = float(row.get("risk_reward", 0.0))
    if risk_reward < cfg.rr_floor:
        return None
    orig_prob = float(row.get("orig_prob", 0.0))
    model_prob = row.get("model_prob")
    model_prob_f = float(model_prob) if model_prob is not None else None
    blended = blend_probability(orig_prob, model_prob_f, cfg.model_weight)
    threshold = max(cfg.min_prob_success, strategy.probability_threshold(risk_reward))
    if blended < threshold:
        return None
    return {
        "symbol": row.get("symbol", "UNKNOWN"),
        "side": row.get("side", "BUY"),
        "probability": blended,
        "risk_reward": risk_reward,
        "timestamp": row.get("timestamp"),
    }


def apply_trade(signal: dict, row: pd.Series, cfg: BacktestConfig) -> TradeResult:
    outcome = float(row.get("outcome", 0.0))
    risk_reward = float(signal["risk_reward"])
    probability = float(signal["probability"])
    slippage = cfg.slippage_bps / 10_000.0
    trade_cost = cfg.fees + slippage
    if outcome > 0:
        profit = risk_reward - trade_cost
    else:
        profit = -1.0 - trade_cost
    timestamp_value = signal["timestamp"]
    ts = pd.Timestamp(timestamp_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return TradeResult(
        timestamp=ts,
        symbol=str(signal["symbol"]),
        side=str(signal["side"]),
        probability=probability,
        risk_reward=risk_reward,
        outcome=outcome,
        profit=profit,
    )


def compute_kpis(trades: Iterable[TradeResult]) -> dict:
    trades_list = list(trades)
    if not trades_list:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }
    profits = np.array([t.profit for t in trades_list])
    wins = profits > 0
    win_rate = float(wins.mean())
    expectancy = float(profits.mean())
    gains = profits[profits > 0].sum()
    losses = -profits[profits < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf" if gains > 0 else 0.0)
    equity = profits.cumsum()
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0
    first_ts = trades_list[0].timestamp
    last_ts = trades_list[-1].timestamp
    years = max((last_ts - first_ts).days / 365.25, 1e-6)
    cumulative_return = float(profits.sum())
    if cumulative_return <= -1.0:
        cagr = float("nan")
    else:
        cagr = float(((1 + cumulative_return) ** (1 / years)) - 1)
    std = profits.std(ddof=1) if len(profits) > 1 else 0.0
    sharpe = float(expectancy / std) if std else 0.0
    return {
        "total_trades": len(trades_list),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def run_backtest(cfg: BacktestConfig, data: pd.DataFrame) -> dict:
    results: list[TradeResult] = []
    for _, test_chunk in _iter_splits(data, cfg):
        for _, row in test_chunk.iterrows():
            signal = simulate_decision(row, cfg)
            if not signal:
                continue
            trade = apply_trade(signal, row, cfg)
            results.append(trade)
    kpis = compute_kpis(results)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(config.BACKTEST_REPORT_DIR) / f"backtest_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    trades_frame = pd.DataFrame([t.__dict__ for t in results])
    trades_frame.to_csv(report_dir / "trades.csv", index=False)
    with (report_dir / "kpis.json").open("w", encoding="utf-8") as handle:
        json.dump(kpis, handle, indent=2, default=str)
    with (report_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg.__dict__, handle, indent=2, default=str)
    return kpis


def _load_config(path: Path) -> BacktestConfig:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load backtest configs") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return BacktestConfig(**payload)


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "timestamp" not in frame.columns:
        raise ValueError("Dataset must contain a 'timestamp' column")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a walk-forward backtest")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file")
    parser.add_argument("--data", type=Path, required=True, help="CSV dataset with features")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    data = _load_dataset(args.data)
    kpis = run_backtest(cfg, data)
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
