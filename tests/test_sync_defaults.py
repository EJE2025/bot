import os
import sys

import pytest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import trading_bot.bot as bot
from trading_bot import config, execution, trade_manager, shutdown


@pytest.fixture(autouse=True)
def _reset_state():
    trade_manager.reset_state()
    shutdown.reset_for_tests()
    yield
    trade_manager.reset_state()
    shutdown.reset_for_tests()


def test_sync_applies_side_specific_fallbacks(monkeypatch):
    positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "contracts": 1,
            "entryPrice": 100.0,
            "side": "long",
            "leverage": 3,
            "stopLossPrice": 0,
            "takeProfitPrice": 0,
        },
        {
            "symbol": "ETH/USDT:USDT",
            "contracts": 2,
            "entryPrice": 50.0,
            "side": "short",
            "leverage": 2,
            "stopLossPrice": 0,
            "takeProfitPrice": 0,
        },
    ]

    monkeypatch.setattr(execution, "exchange", object())
    monkeypatch.setattr(bot.permissions, "audit_environment", lambda ex: None)
    monkeypatch.setattr(bot, "load_trades", lambda: None)
    monkeypatch.setattr(bot, "save_trades", lambda: None)
    monkeypatch.setattr(bot.execution, "fetch_positions", lambda: positions)
    monkeypatch.setattr(bot.execution, "fetch_open_orders", lambda: [])
    monkeypatch.setattr(bot.execution, "cancel_order", lambda *a, **k: None)
    monkeypatch.setattr(bot.execution, "cleanup_old_orders", lambda: None)
    monkeypatch.setattr(bot.execution, "cancel_all_orders", lambda: None)
    monkeypatch.setattr(bot, "reconcile_pending_trades", lambda: None)
    monkeypatch.setattr(bot.Thread, "start", lambda self: None)
    monkeypatch.setattr(bot, "start_metrics_server", lambda *a, **k: None)
    monkeypatch.setattr(bot, "monitor_system", lambda *a, **k: None)
    monkeypatch.setattr(bot.strategy, "start_liquidity", lambda symbols=None: None)
    monkeypatch.setattr(bot.new_dashboard, "start_dashboard", lambda host, port: None)
    monkeypatch.setattr(bot.auto_trainer, "start_auto_trainer", lambda stop_event: None)
    monkeypatch.setattr(bot.shutdown, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(bot.shutdown, "register_callback", lambda cb: None)
    monkeypatch.setattr(bot.shutdown, "execute_callbacks", lambda: None)
    monkeypatch.setattr(bot, "maybe_reload_model", lambda *a, **k: None)
    monkeypatch.setattr(bot, "current_model", lambda: None)
    monkeypatch.setattr(bot.auto_trainer, "observe_live_metrics", lambda metrics: None)
    monkeypatch.setattr(
        bot.MODEL_MONITOR,
        "metrics",
        lambda: {
            "count": 0,
            "hit_rate": None,
            "p_value": None,
        },
        raising=False,
    )
    monkeypatch.setattr(bot, "maybe_alert", lambda *a, **k: None)
    monkeypatch.setattr(bot.notify, "send_telegram", lambda *a, **k: None)
    monkeypatch.setattr(bot.notify, "send_discord", lambda *a, **k: None)

    monkeypatch.setattr(config, "ENABLE_TRADING", False)
    monkeypatch.setattr(config, "AUTO_TRADE", False)
    monkeypatch.setattr(config, "SHADOW_MODE", False)
    monkeypatch.setattr(config, "MAINTENANCE", False)
    monkeypatch.setattr(config, "KILL_SWITCH_ON_DRIFT", False)
    monkeypatch.setattr(config, "LOOP_INTERVAL", 1)
    monkeypatch.setattr(config, "RECONCILE_INTERVAL_S", 10_000)
    monkeypatch.setattr(config, "TRAILING_STOP_ENABLED", False)

    monkeypatch.setattr(bot.data, "get_current_price_ticker", lambda symbol: None)

    def interrupt_sleep(_seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(bot.time, "sleep", interrupt_sleep)

    bot.run()

    open_trades = {trade["symbol"]: trade for trade in trade_manager.all_open_trades()}

    long_trade = open_trades["BTC_USDT"]
    short_trade = open_trades["ETH_USDT"]

    assert long_trade["side"] == "BUY"
    assert short_trade["side"] == "SELL"

    assert long_trade["stop_loss"] == pytest.approx(98.0)
    assert long_trade["take_profit"] == pytest.approx(102.0)

    assert short_trade["stop_loss"] == pytest.approx(51.0)
    assert short_trade["take_profit"] == pytest.approx(49.0)
