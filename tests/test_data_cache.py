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

