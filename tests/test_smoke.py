# tests/test_smoke.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def test_open_close_trade_smoke(monkeypatch):
    import trading_bot.bot as bot
    import trading_bot.trade_manager as tm
    from trading_bot import execution

    signal = {
        "symbol": "BTC_USDT",
        "side": "BUY",
        "quantity": 1.0,
        "entry_price": 100.0,
        "take_profit": 110.0,
        "stop_loss": 90.0,
        "leverage": 5,
    }
    monkeypatch.setattr(bot, "save_trades", lambda: None)
    monkeypatch.setattr(execution, "open_position", lambda *a, **k: {"id": "1", "average": 100.0})
    monkeypatch.setattr(execution, "fetch_order_status", lambda oid, sym: "filled")
    monkeypatch.setattr(
        execution,
        "get_order_fill_details",
        lambda oid, sym: {"filled": 1.0, "remaining": 0.0, "average": 100.0},
    )
    trade = bot.open_new_trade(signal)
    assert trade is not None
    assert tm.count_open_trades() == 1

    monkeypatch.setattr(execution, "close_position", lambda *a, **k: {"id": "2", "average": 105.0})
    monkeypatch.setattr(execution, "fetch_order_status", lambda oid, sym: "filled")
    monkeypatch.setattr(
        execution,
        "get_order_fill_details",
        lambda oid, sym: {"filled": 1.0, "remaining": 0.0, "average": 105.0},
    )
    monkeypatch.setattr(execution, "fetch_position_size", lambda sym: 0.0)

    closed, exec_price, realized = bot.close_existing_trade(trade, reason="TP")
    assert closed is not None
    assert exec_price is not None
    assert realized is not None
    assert tm.count_open_trades() == 0
    assert len(tm.all_closed_trades()) == 1

