import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import trade_manager as tm


def test_add_trade_unique_id():
    tm.reset_state()
    tm.add_trade({"symbol": "BTC_USDT"})

    with pytest.raises(ValueError):
        tm.add_trade({"symbol": "BTC_USDT"})

    tm.add_trade({"symbol": "BTC_USDT"}, allow_duplicates=True)
    ids = [t["trade_id"] for t in tm.all_open_trades()]
    assert len(ids) == len(set(ids))
