import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot.utils import normalize_symbol


def test_normalize_symbol():
    assert normalize_symbol("BTC/USDT") == "BTCUSDT"
    assert normalize_symbol("ETH_USDT") == "ETHUSDT"

