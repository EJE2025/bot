import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot.utils import normalize_symbol
from trading_bot import trade_manager as tm


def test_normalize_symbol():
    assert normalize_symbol("btc/usdt") == "BTC_USDT"
    assert normalize_symbol("BTCUSDT") == "BTC_USDT"
    assert normalize_symbol("ETH/USDT:USDT") == "ETH_USDT"


def test_find_trade_normalized(monkeypatch):
    tm.open_trades.clear()
    tm.add_trade({"symbol": "BTC_USDT"})
    tid = tm.all_open_trades()[0]["trade_id"]
    found = tm.find_trade(symbol="btc/usdt:usdt")
    assert found and found["trade_id"] == tid
