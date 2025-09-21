import os
import sys
from datetime import datetime, timedelta, timezone
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


def test_trade_age_minutes_roundtrip():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    opened = (now - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    trade = {"open_time": opened}
    age = tm.trade_age_minutes(trade, now=now)
    assert age is not None
    assert pytest.approx(age, rel=1e-3) == 30


def test_exceeded_max_duration_detection():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trade = {
        "open_time": (now - timedelta(minutes=120)).isoformat().replace("+00:00", "Z"),
        "max_duration_minutes": 90,
    }
    assert tm.exceeded_max_duration(trade, now=now)
    trade["max_duration_minutes"] = 180
    assert not tm.exceeded_max_duration(trade, now=now)

