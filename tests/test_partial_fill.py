import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import trading_bot.bot as bot
from trading_bot import trade_manager, execution
from trading_bot.exchanges import MockExchange


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

    monkeypatch.setattr(execution, "open_position", lambda *a, **k: {"id": "1"})
    monkeypatch.setattr(execution, "check_order_filled", lambda oid, sym: False)
    monkeypatch.setattr(execution, "cancel_order", lambda oid, sym: None)

    def fake_fetch_positions():
        return [{
            "symbol": "AAA/USDT:USDT",
            "contracts": 4,
            "entryPrice": 1.01,
            "side": "long",
            "leverage": 1,
        }]

    monkeypatch.setattr(execution, "fetch_positions", fake_fetch_positions)
    monkeypatch.setattr(bot, "save_trades", lambda: None)

    trade = bot.open_new_trade(signal)
    assert trade is not None
    assert trade["quantity"] == 4
    assert trade_manager.count_open_trades() == 1
    assert trade_manager.find_trade(symbol="AAA_USDT") is not None
