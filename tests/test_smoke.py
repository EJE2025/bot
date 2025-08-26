# tests/test_smoke.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def test_open_close_trade_smoke():
    import trading_bot.bot as bot
    import trading_bot.trade_manager as tm

    signal = {
        "symbol": "BTC_USDT",
        "side": "BUY",
        "quantity": 1.0,
        "entry_price": 100.0,
        "take_profit": 110.0,
        "stop_loss": 90.0,
        "leverage": 5,
    }
    trade = bot.open_new_trade(signal)
    assert trade is not None
    assert tm.count_open_trades() == 1

    bot.close_existing_trade(trade, exit_price=105.0, profit=5.0, reason="TP")
    assert tm.count_open_trades() == 0
    assert len(tm.all_closed_trades()) == 1

