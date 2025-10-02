import math

import pytest

from trading_bot import trade_manager, webapp


@pytest.fixture(autouse=True)
def reset_state():
    trade_manager.reset_state()
    yield
    trade_manager.reset_state()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(webapp.data, "get_current_price_ticker", lambda *_: None)
    app = webapp.app
    app.config.update(TESTING=True)
    return app.test_client()


def _open_and_close(symbol: str, side: str, entry: float, qty: float, exit_price: float) -> None:
    trade = trade_manager.add_trade(
        {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "quantity": qty,
        }
    )
    trade_manager.close_trade_full(trade["trade_id"], exit_price=exit_price)


def test_summary_includes_realized_metrics(client):
    _open_and_close("BTC_USDT", "BUY", entry=2.0, qty=1.0, exit_price=3.0)
    _open_and_close("ETH_USDT", "SELL", entry=4.0, qty=1.0, exit_price=3.0)
    _open_and_close("SOL_USDT", "BUY", entry=10.0, qty=1.0, exit_price=8.0)

    response = client.get("/api/summary")
    assert response.status_code == 200

    payload = response.get_json()
    assert math.isclose(payload["realized_pnl"], 0.0, abs_tol=1e-9)
    assert math.isclose(payload["realized_balance"], 14.0, rel_tol=1e-9)
