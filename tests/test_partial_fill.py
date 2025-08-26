import os
import sys

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
    monkeypatch.setattr(execution, "fetch_order_status", lambda oid, sym: "partial")
    monkeypatch.setattr(bot, "save_trades", lambda: None)

    trade = bot.open_new_trade(signal)
    assert trade is not None
    stored = trade_manager.find_trade(symbol="AAA_USDT")
    assert stored["state"] == TradeState.PARTIALLY_FILLED.value
    assert trade_manager.count_open_trades() == 1
