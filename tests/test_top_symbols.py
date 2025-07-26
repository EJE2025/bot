import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import data, config
from trading_bot.exchanges import MockExchange


def test_common_symbols_filters_blacklist(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(config, "TEST_MODE", False)
    monkeypatch.setattr(config, "BLACKLIST_SYMBOLS", {"BTCUSDT"})
    syms = data.get_common_top_symbols(ex, 5)
    assert "BTCUSDT" not in syms
    assert len(syms) == 5


def test_common_symbols_exclude(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(config, "TEST_MODE", False)
    syms = data.get_common_top_symbols(ex, 5, exclude=["eth_usdt"])
    assert "ETHUSDT" not in syms
    assert len(syms) == 5


def test_common_symbols_test_mode(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(config, "TEST_MODE", True)
    monkeypatch.setattr(config, "TEST_SYMBOLS", ["FOOUSDT", "BARUSDT"])
    syms = data.get_common_top_symbols(ex, 2)
    assert syms == ["FOOUSDT", "BARUSDT"]
