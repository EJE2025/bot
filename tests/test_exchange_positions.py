import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trading_bot.exchanges import MockExchange


def test_open_long_then_close_long():
    ex = MockExchange()
    ex.create_order("BTC/USDT:USDT", "market", "buy", 1)
    assert len(ex.fetch_positions()) == 1
    ex.create_order("BTC/USDT:USDT", "market", "close_long", 1)
    assert ex.fetch_positions() == []
    bal = ex.fetch_balance()["USDT"]
    assert bal["free"] == 100_000
    assert bal["used"] == 0


def test_open_short_then_close_short():
    ex = MockExchange()
    ex.create_order("BTC/USDT:USDT", "market", "sell", 1)
    assert len(ex.fetch_positions()) == 1
    ex.create_order("BTC/USDT:USDT", "market", "close_short", 1)
    assert ex.fetch_positions() == []
    bal = ex.fetch_balance()["USDT"]
    assert bal["free"] == 100_000
    assert bal["used"] == 0
