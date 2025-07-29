import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import trading_bot.bot as bot
from trading_bot import config, trade_manager, execution, data
from trading_bot.exchanges import MockExchange


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    # use mock exchange
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    trade_manager.reset_state()
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
