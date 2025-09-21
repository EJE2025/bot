import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import backtest
from trading_bot import config


def test_backtest_generates_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKTEST_REPORT_DIR", str(tmp_path))
    timestamps = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    outcomes = [1 if i % 2 == 0 else -1 for i in range(20)]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 20,
            "side": ["BUY"] * 20,
            "orig_prob": [0.6 + i * 0.01 for i in range(20)],
            "model_prob": [0.65 + i * 0.01 for i in range(20)],
            "risk_reward": [2.0] * 20,
            "outcome": outcomes,
        }
    )
    cfg = backtest.BacktestConfig(
        mode="walk_forward",
        train_start="2024-01-01",
        test_start="2024-01-05",
        test_end="2024-01-20",
        rolling_train_days=5,
        rolling_test_days=5,
        fees=0.01,
        slippage_bps=5,
    )
    metrics = backtest.run_backtest(cfg, frame)
    assert metrics["total_trades"] > 0
    report_dirs = list(tmp_path.glob("backtest_*/kpis.json"))
    assert report_dirs, "Expected KPI report file"
    with report_dirs[0].open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["total_trades"] == metrics["total_trades"]


def test_iter_splits_respect_temporal_order():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 10,
            "side": ["BUY"] * 10,
            "orig_prob": [0.6] * 10,
            "risk_reward": [2.0] * 10,
            "outcome": [1] * 10,
        }
    )
    cfg = backtest.BacktestConfig(
        mode="rolling",
        train_start="2024-01-01",
        test_start="2024-01-05",
        test_end="2024-01-10",
        rolling_train_days=3,
        rolling_test_days=2,
    )
    splits = list(backtest._iter_splits(frame, cfg))
    assert splits
    for train, test in splits:
        assert train["timestamp"].max() < test["timestamp"].min()
