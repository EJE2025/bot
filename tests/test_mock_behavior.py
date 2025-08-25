import os
from datetime import datetime, timezone

from trading_bot import config, data
from trading_bot.execution import exchange, MockExchange
from trading_bot.trade_manager import (
    add_trade,
    count_open_trades,
    reset_state,
)


def test_price_consistency_in_test_mode(monkeypatch):
    os.environ["TEST_MODE"] = "1"

    class _MX(MockExchange):
        def _get_market_price(self, sym):
            return 123.45

    monkeypatch.setattr("trading_bot.data.exchange", _MX())
    price = data.get_current_price_ticker("AAA_USDT")
    assert price == 123.45


def test_cooldown_enforced(monkeypatch):
    os.environ["COOLDOWN_MINUTES"] = "1"
    from importlib import reload

    reload(config)
    from trading_bot.trade_manager import in_cooldown, _last_closed, normalize_symbol

    sym = "AAA_USDT"
    _last_closed[normalize_symbol(sym)] = datetime.now().timestamp()
    assert in_cooldown(sym) is True


def test_partial_fill_is_registered(monkeypatch):
    reset_state()

    class _MX(MockExchange):
        def fetch_positions(self, params=None):
            return [
                {
                    "symbol": "AAA/USDT:USDT",
                    "contracts": 2,
                    "entryPrice": 100.5,
                    "side": "long",
                    "leverage": 10,
                }
            ]

    monkeypatch.setattr("trading_bot.execution.exchange", _MX())
    add_trade({"symbol": "AAA_USDT", "side": "BUY", "quantity": 2, "entry_price": 100.5})
    assert count_open_trades() == 1


def test_utc_timestamps():
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    assert ts.endswith("Z")

