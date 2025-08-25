import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import trading_bot.config as config
import trading_bot.trade_manager as tm
from trading_bot.state_machine import TradeState, InvalidStateTransition

tm.reset_state()

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


def test_set_trade_state(monkeypatch):
    tm.reset_state()
    tm.add_trade({"symbol": "BTCUSDT"})
    t = tm.all_open_trades()[0]
    tid = t["trade_id"]
    tm.set_trade_state(tid, TradeState.OPEN)
    assert t["state"] == TradeState.OPEN.value
    with pytest.raises(InvalidStateTransition):
        tm.set_trade_state(tid, TradeState.PENDING)


def test_close_trade_forces_closing(monkeypatch):
    tm.reset_state()
    tm.add_trade({"symbol": "ETHUSDT"})
    t = tm.all_open_trades()[0]
    tid = t["trade_id"]
    tm.set_trade_state(tid, TradeState.OPEN)

    calls = []
    original = tm.set_trade_state

    def wrapper(trade_id, new_state):
        calls.append(new_state)
        return original(trade_id, new_state)

    monkeypatch.setattr(tm, "set_trade_state", wrapper)

    tm.close_trade(trade_id=tid)

    assert calls == [TradeState.CLOSING, TradeState.CLOSED]
    assert tm.find_trade(trade_id=tid) is None
    assert tm.closed_trades[-1]["state"] == TradeState.CLOSED.value

