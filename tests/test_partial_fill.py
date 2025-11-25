import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import trading_bot.bot as bot
from trading_bot import trade_manager, execution
from trading_bot.exchanges import MockExchange
from trading_bot.state_machine import TradeState


def test_partial_fill_is_registered(monkeypatch):
    ex = MockExchange(order_status_flow="open")
    monkeypatch.setattr(execution, "exchange", ex)
    trade_manager.reset_state()

    signal = {
        "symbol": "AAA_USDT",
        "side": "BUY",
        "quantity": 10,
        "entry_price": 1.0,
        "take_profit": 1.1,
        "stop_loss": 0.9,
        "prob_success": 0.8,
        "risk_reward": 2,
        "leverage": 1,
    }

    monkeypatch.setattr(
        execution,
        "open_position",
        lambda *a, **k: {"id": "1", "average": signal["entry_price"]},
    )
    monkeypatch.setattr(bot, "save_trades", lambda: None)

    initial_qty = signal["quantity"]
    trade = bot.open_new_trade(signal)
    assert trade is not None
    trade_manager.ws_order_partial({"symbol": "AAAUSDT", "filledSize": 4.0, "avgPrice": 1.0})
    stored = trade_manager.find_trade(symbol="AAA_USDT")
    assert stored["state"] == TradeState.PARTIALLY_FILLED.value
    assert stored["quantity"] == 4.0
    assert stored.get("quantity_remaining") == pytest.approx(6.0)
    assert stored.get("requested_quantity") == initial_qty
    assert trade_manager.count_open_trades() == 1


def test_partial_close_retries_remaining(monkeypatch):
    ex = MockExchange(order_status_flow="open")
    monkeypatch.setattr(execution, "exchange", ex)
    trade_manager.reset_state()

    trade = {
        "trade_id": "T2",
        "symbol": "BBB_USDT",
        "side": "BUY",
        "quantity": 1.0,
        "entry_price": 100.0,
        "take_profit": 110.0,
        "stop_loss": 95.0,
        "prob_success": 0.7,
        "leverage": 1,
    }

    monkeypatch.setattr(bot, "save_trades", lambda: None)
    trade_manager.add_trade(trade)
    trade_manager.set_trade_state(trade["trade_id"], TradeState.OPEN)

    close_calls: list[float] = []

    def _close_position(symbol, side, amount, order_type="market"):
        close_calls.append(amount)
        return {"id": "1", "average": 100.0}

    monkeypatch.setattr(execution, "close_position", _close_position)
    monkeypatch.setattr(trade_manager.data, "get_current_price_ticker", lambda symbol: 100.0)

    closed, exec_price, realized = bot.close_existing_trade(trade, reason="SL")
    assert closed is None
    assert exec_price == pytest.approx(100.0)
    assert realized is None
    assert close_calls and close_calls[0] == pytest.approx(1.0)

    trade_manager.ws_position_closed({"symbol": "BBBUSDT", "holdVolume": 0, "avgPrice": 100.0})
    assert trade_manager.all_closed_trades()
