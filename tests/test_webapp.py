from trading_bot.webapp import app
from trading_bot import webapp as webapp_module
from trading_bot.trade_manager import open_trades, closed_trades


def test_webapp_endpoints(tmp_path, monkeypatch):
    """Prueba que los endpoints /api/trades y /api/liquidity devuelven datos válidos."""
    open_trades.clear()
    closed_trades.clear()

    open_trades.append({
        "symbol": "BTC_USDT",
        "entry_price": 100.0,
        "quantity": 1.0,
        "side": "BUY",
    })

    history_file = tmp_path / "trade_history.csv"
    history_file.write_text(
        "symbol,side,quantity,entry_price,exit_price,take_profit,stop_loss,profit,open_time,close_time\n"
        "BTC_USDT,BUY,1.0,100.0,110.0,0,0,10.0,2024-01-01T00:00:00,2024-01-01T01:00:00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_module, "HISTORY_FILE", history_file)

    with app.test_client() as client:
        resp = client.get("/api/trades")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert data[0]["symbol"] == "BTC_USDT"

        resp2 = client.get("/api/liquidity")
        assert resp2.status_code == 200
        assert isinstance(resp2.get_json(), dict)

        summary_resp = client.get("/api/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.get_json()
        assert summary["total_positions"] == 1
        assert summary["per_symbol"][0]["symbol"] == "BTC_USDT"

        history_resp = client.get("/api/history")
        assert history_resp.status_code == 200
        history_data = history_resp.get_json()
        assert isinstance(history_data, list)
        assert history_data[0]["symbol"] == "BTC_USDT"
