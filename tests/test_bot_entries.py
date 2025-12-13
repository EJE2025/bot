import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import trading_bot.bot as bot
from trading_bot import config, trade_manager, execution, data
import trading_bot.strategy as strategy
from trading_bot.exchanges import MockExchange


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    # use mock exchange
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    trade_manager.reset_state()
    monkeypatch.setattr(config, "MIN_POSITION_SIZE_USDT", 0.0)
    monkeypatch.setattr(strategy.rl_agent, "adjust_targets", lambda *args, **kwargs: None)
    yield
    trade_manager.reset_state()


def test_open_multiple_trades(monkeypatch):
    monkeypatch.setattr(config, 'MAX_OPEN_TRADES', 3)
    monkeypatch.setattr(config, 'MIN_RISK_REWARD', 0)
    monkeypatch.setattr(bot.data, 'get_common_top_symbols', lambda ex, n: ['AAA_USDT','BBB_USDT','CCC_USDT'])
    monkeypatch.setattr(data, 'get_common_top_symbols', lambda ex, n: ['AAA_USDT','BBB_USDT','CCC_USDT'])
    signals = {
        'AAA_USDT': {'symbol':'AAA_USDT','side':'BUY','quantity':1,'entry_price':1,'take_profit':1.1,'stop_loss':0.9,'prob_success':0.8,'risk_reward':2,'leverage':1},
        'BBB_USDT': {'symbol':'BBB_USDT','side':'BUY','quantity':1,'entry_price':2,'take_profit':2.2,'stop_loss':1.8,'prob_success':0.8,'risk_reward':2,'leverage':1},
        'CCC_USDT': {'symbol':'CCC_USDT','side':'BUY','quantity':1,'entry_price':3,'take_profit':3.3,'stop_loss':2.7,'prob_success':0.8,'risk_reward':2,'leverage':1},
    }
    monkeypatch.setattr(bot.strategy, 'decidir_entrada', lambda sym, modelo_historico=None: signals[sym])
    bot.run_one_iteration_open()
    assert trade_manager.count_open_trades() == 3




def test_trade_cooldown(monkeypatch):
    monkeypatch.setattr(config, 'MAX_OPEN_TRADES', 1)
    monkeypatch.setattr(config, 'MIN_RISK_REWARD', 0)
    monkeypatch.setattr(config, 'TRADE_COOLDOWN', 60)

    monkeypatch.setattr(bot.data, 'get_common_top_symbols', lambda ex, n: ['AAA_USDT'])
    monkeypatch.setattr(data, 'get_common_top_symbols', lambda ex, n: ['AAA_USDT'])
    signal = {'symbol':'AAA_USDT','side':'BUY','quantity':1,'entry_price':1,'take_profit':1.1,'stop_loss':0.9,'prob_success':0.8,'risk_reward':2,'leverage':1}
    monkeypatch.setattr(bot.strategy, 'decidir_entrada', lambda sym, modelo_historico=None: signal)

    # open first trade
    bot.run_one_iteration_open()
    assert trade_manager.count_open_trades() == 1

    # close trade and keep cooldown timestamp
    trade_manager.close_trade(symbol='AAA_USDT')
    assert trade_manager.count_open_trades() == 0

    # before cooldown expires should not open again
    bot.run_one_iteration_open()
    assert trade_manager.count_open_trades() == 0


def test_run_one_iteration_respects_min_notional(monkeypatch):
    monkeypatch.setattr(config, "MIN_POSITION_SIZE_USDT", 10.0)
    monkeypatch.setattr(config, "MIN_RISK_REWARD", 0)
    monkeypatch.setattr(bot.data, "get_common_top_symbols", lambda ex, n: ["AAA_USDT"])
    monkeypatch.setattr(data, "get_common_top_symbols", lambda ex, n: ["AAA_USDT"])
    signal = {
        "symbol": "AAA_USDT",
        "side": "BUY",
        "quantity": 1,
        "entry_price": 5,
        "take_profit": 6,
        "stop_loss": 4,
        "prob_success": 0.8,
        "risk_reward": 2,
        "leverage": 1,
    }
    monkeypatch.setattr(bot.strategy, "decidir_entrada", lambda sym, modelo_historico=None: signal)

    bot.run_one_iteration_open()

    assert trade_manager.count_open_trades() == 0


def test_stop_loss_respects_max_pct(monkeypatch):
    base_prices = [100 + i * 0.1 for i in range(20)]
    entry_price = base_prices[-1]
    info = {
        "close": base_prices,
        "high": [p + 1 for p in base_prices],
        "low": [p - 1 for p in base_prices],
        "vol": [1000 for _ in base_prices],
    }

    monkeypatch.setattr(config, "STOP_ATR_MULT", 2.0)
    monkeypatch.setattr(config, "MAX_STOP_LOSS_PCT", 0.05)

    monkeypatch.setattr(strategy, "calculate_support_resistance", lambda closes: (None, None))
    monkeypatch.setattr(strategy, "compute_rsi", lambda closes, period: np.array([40.0]))
    monkeypatch.setattr(
        strategy, "compute_macd", lambda closes: (np.array([1.0]), np.array([0.0]), np.array([0.0]))
    )
    monkeypatch.setattr(
        strategy,
        "calculate_atr",
        lambda highs, lows, closes: entry_price,
    )
    monkeypatch.setattr(strategy, "sentiment_score", lambda symbol: 0.0)
    monkeypatch.setattr(
        strategy.data,
        "get_order_book",
        lambda symbol: {"bids": [(entry_price - 1, 1)], "asks": [(entry_price + 1, 1)]},
    )
    monkeypatch.setattr(strategy.data, "order_book_imbalance", lambda book, price: 1.0)
    monkeypatch.setattr(
        strategy.data,
        "top_liquidity_levels",
        lambda book: ([(entry_price - 1, 1)], [(entry_price + 1, 1)]),
    )
    monkeypatch.setattr(strategy, "position_sizer", lambda symbol, features, ctx: entry_price)
    monkeypatch.setattr(strategy.execution, "fetch_balance", lambda: 1000.0)
    monkeypatch.setattr(
        strategy,
        "passes_probability_threshold",
        lambda prob, risk_reward, volatility=None: True,
    )
    monkeypatch.setattr(strategy, "probability_threshold", lambda risk_reward, volatility=None: 0.0)
    monkeypatch.setattr(strategy, "log_signal_details", lambda *args, **kwargs: None)

    signal = strategy.decidir_entrada("BTC/USDT", info=info)
    assert signal is not None
    expected_stop = entry_price * (1 - config.MAX_STOP_LOSS_PCT)
    assert signal["stop_loss"] == pytest.approx(expected_stop)
