from trading_bot.webapp import app
from trading_bot.trade_manager import open_trades, closed_trades


def test_webapp_endpoints():
    """Prueba que los endpoints /api/trades y /api/liquidity devuelven datos válidos."""
    open_trades.clear()
    closed_trades.clear()

    open_trades.append({
        "symbol": "BTC_USDT",
        "entry_price": 100.0,
        "quantity": 1.0,
        "side": "BUY",
    })

    with app.test_client() as client:
        resp = client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert data[0]["symbol"] == "BTC_USDT"

        resp2 = client.get("/api/liquidity")
        assert resp2.status_code == 200
        assert isinstance(resp2.get_json(), dict)
