import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import data, config, execution
from trading_bot.exchanges import MockExchange


def test_price_consistency_in_test_mode(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    monkeypatch.setattr(data, "exchange", ex)
    monkeypatch.setattr(config, "TEST_MODE", True)
    monkeypatch.setattr(ex, "_get_market_price", lambda s: 123.45)
    price = data.get_current_price_ticker("AAA_USDT")
    assert price == 123.45
