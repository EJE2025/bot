import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import trading_bot.bot as bot
from trading_bot import config, trade_manager, execution, data
from trading_bot.exchanges import MockExchange


def test_cooldown_enforced_in_run_loop(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    trade_manager.reset_state()

    monkeypatch.setattr(config, "MAX_OPEN_TRADES", 1)
    monkeypatch.setattr(config, "MIN_RISK_REWARD", 0)
    monkeypatch.setattr(config, "TRADE_COOLDOWN", 60)
    monkeypatch.setattr(config, "TEST_MODE", True)

    monkeypatch.setattr(data, "get_common_top_symbols", lambda exg, n: ["AAA_USDT"])
    signal = {
        "symbol": "AAA_USDT",
        "side": "BUY",
        "quantity": 1,
        "entry_price": 1.0,
        "take_profit": 1.1,
        "stop_loss": 0.9,
        "prob_success": 0.8,
        "risk_reward": 2,
        "leverage": 1,
    }
    monkeypatch.setattr(bot.strategy, "decidir_entrada", lambda sym, modelo_historico=None: signal)

    # avoid starting threads or network services
    monkeypatch.setattr(bot.Thread, "start", lambda self: None)
    monkeypatch.setattr(bot, "start_metrics_server", lambda *a, **k: None)
    monkeypatch.setattr(bot, "monitor_system", lambda *a, **k: None)
    monkeypatch.setattr(bot.strategy, "start_liquidity", lambda symbols=None: None)
    monkeypatch.setattr(bot.webapp, "start_dashboard", lambda host, port: None)
    monkeypatch.setattr(bot.optimizer, "load_model", lambda path: None)
    monkeypatch.setattr(bot.trade_manager, "save_trades", lambda: None)
    monkeypatch.setattr(bot.notify, "send_telegram", lambda *a, **k: None)
    monkeypatch.setattr(bot.notify, "send_discord", lambda *a, **k: None)

    calls = {"sleep": 0}

    def fake_sleep(sec):
        calls["sleep"] += 1
        if calls["sleep"] == 1:
            trade_manager.close_trade(symbol="AAA_USDT")
            return
        raise KeyboardInterrupt()

    monkeypatch.setattr(bot.time, "sleep", fake_sleep)

    bot.run()

    assert trade_manager.count_open_trades() == 0
    assert len(trade_manager.all_closed_trades()) == 1
