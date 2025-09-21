import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import trading_bot.bot as bot
import trading_bot.trade_manager as tm
from trading_bot.state_machine import TradeState
from trading_bot import execution
from trading_bot.exchanges import MockExchange


def test_pending_to_open_and_close(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    tm.reset_state()

    signal = {
        "trade_id": "T1",
        "symbol": "BTC_USDT",
        "side": "BUY",
        "quantity": 1.0,
        "entry_price": 100.0,
        "take_profit": 110.0,
        "stop_loss": 90.0,
        "leverage": 1,
    }

    monkeypatch.setattr(bot, "save_trades", lambda: None)
    monkeypatch.setattr(execution, "open_position", lambda *a, **k: {"id": "1", "average": 100.0})
    monkeypatch.setattr(execution, "fetch_order_status", lambda oid, sym: "filled")
    monkeypatch.setattr(
        execution,
        "get_order_fill_details",
        lambda oid, sym: {"filled": 1.0, "remaining": 0.0, "average": 100.0},
    )

    tr = bot.open_new_trade(signal)
    assert tm.find_trade(trade_id="T1")["state"] == TradeState.OPEN.value

    monkeypatch.setattr(execution, "close_position", lambda *a, **k: {"id": "2", "average": 105.0})
    monkeypatch.setattr(execution, "fetch_order_status", lambda oid, sym: "filled")
    monkeypatch.setattr(execution, "get_order_fill_details", lambda oid, sym: {"filled": 1.0, "remaining": 0.0, "average": 105.0})
    monkeypatch.setattr(execution, "fetch_position_size", lambda sym: 0.0)

    closed, exec_price, realized = bot.close_existing_trade(tr, reason="TP")
    assert closed is not None
    assert exec_price == 105.0
    assert realized == 5.0
    assert len(tm.all_closed_trades()) == 1
    assert tm.all_closed_trades()[0]["state"] == TradeState.CLOSED.value
