import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import trading_bot.config as config
import trading_bot.trade_manager as tm


def test_history_size_limit(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TRADE_HISTORY_LOG", True)
    monkeypatch.setattr(config, "MAX_TRADE_HISTORY_SIZE", 2)

    tm.add_trade({"symbol": "BTCUSDT"})
    tid = tm.all_open_trades()[0]["trade_id"]
    tm.update_trade(tid, foo=1)
    tm.close_trade(trade_id=tid)
    assert len(tm.get_history()) == 2


def test_atomic_save(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TRADE_HISTORY_LOG", False)
    tm.add_trade({"symbol": "ETHUSDT"})
    tm.save_trades(tmp_path / "open.json", tmp_path / "closed.json")
    assert (tmp_path / "open.json").exists()
    assert not Path(str(tmp_path / "open.json") + ".tmp").exists()


def test_count_trades_for_symbol():
    tm.add_trade({"symbol": "BTC_USDT"})
    tm.add_trade({"symbol": "BTC_USDT"})
    assert tm.count_trades_for_symbol("BTC_USDT") == 2

