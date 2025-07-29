import json
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import data


def test_get_market_data_uses_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(data, "CACHE_DIR", cache_dir.as_posix())

    symbol = "BTC_USDT"
    interval = "Min15"
    limit = 5
    sample = {"close": ["1"], "high": ["2"], "low": ["0"], "vol": ["10"]}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_dir / f"{symbol.replace('_','')}_{interval}_{limit}.json", "w", encoding="utf-8") as fh:
        json.dump(sample, fh)

    def fail(*args, **kwargs):
        raise RuntimeError("net down")

    monkeypatch.setattr(data.requests, "get", fail)
    monkeypatch.setattr(data.time, "sleep", lambda x: None)

    result = data.get_market_data(symbol, interval=interval, limit=limit)
    assert result == sample


def test_get_market_data_local_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "SAMPLE_DATA_DIR", tmp_path.as_posix())
    csv_path = tmp_path / "BTC_USDT_Min15.csv"
    csv_path.write_text("close,high,low,vol\n1,2,0,10\n2,3,1,20\n")
    monkeypatch.setattr(data.config, "TEST_MODE", False)
    result = data.get_market_data("BTC_USDT", interval="Min15", limit=2)
    assert result == {"close": [1, 2], "high": [2, 3], "low": [0, 1], "vol": [10, 20]}


def test_get_market_data_test_mode(monkeypatch):
    monkeypatch.setattr(data.config, "TEST_MODE", True)
    out1 = data.get_market_data("BTC_USDT", limit=5)
    out2 = data.get_market_data("BTC_USDT", limit=5)
    assert out1 == out2
    assert set(out1) == {"close", "high", "low", "vol"}
    assert len(out1["close"]) == 5

