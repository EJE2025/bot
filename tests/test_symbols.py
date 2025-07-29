import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import data
from trading_bot.exchanges import MockExchange


EXPECTED_TOP = [
    "DOT_USDT", "AVAX_USDT", "MATIC_USDT", "LTC_USDT", "BCH_USDT",
]


def test_mockexchange_deterministic_volume():
    ex1 = MockExchange()
    ex2 = MockExchange()
    assert data.get_common_top_symbols(ex1, 3) == EXPECTED_TOP[:3]
    assert data.get_common_top_symbols(ex2, 3) == EXPECTED_TOP[:3]


def test_get_common_top_symbols_exclude():
    ex = MockExchange()
    top = data.get_common_top_symbols(ex, 5)
    assert top == EXPECTED_TOP

    filtered = data.get_common_top_symbols(ex, 5, exclude=["AVAXUSDT", "LTCUSDT"])
    assert filtered[0] == "DOT_USDT"
    assert "AVAX_USDT" not in filtered
    assert "LTC_USDT" not in filtered
