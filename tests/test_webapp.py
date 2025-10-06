import json
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest
from werkzeug.serving import make_server

from trading_bot.webapp import app
from trading_bot import webapp as webapp_module
from trading_bot.trade_manager import open_trades, closed_trades


def test_webapp_endpoints(tmp_path, monkeypatch):
    """Prueba que los endpoints principales devuelven datos válidos."""
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


def test_history_endpoint_limits_to_last_50(tmp_path, monkeypatch):
    open_trades.clear()
    closed_trades.clear()

    rows = [
        "symbol,side,quantity,entry_price,exit_price,take_profit,stop_loss,profit,open_time,close_time"
    ]
    for idx in range(60):
        rows.append(
            f"BTC_USDT,BUY,1.0,100.0,105.0,0,0,5.0,2024-01-01T00:00:{idx:02d},2024-01-01T01:00:{idx:02d}"
        )

    history_file = tmp_path / "history_large.csv"
    history_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(webapp_module, "HISTORY_FILE", history_file)

    with app.test_client() as client:
        response = client.get("/api/history")
        assert response.status_code == 200
        payload = response.get_json()
        assert isinstance(payload, list)
        assert len(payload) == 50
        first_entry = payload[0]
        assert first_entry["close_time"].endswith("59")


def test_index_same_origin_when_gateway_missing(monkeypatch):
    """La plantilla debe usar el mismo origen cuando no hay gateway configurado."""

    monkeypatch.setattr(webapp_module.config, "DASHBOARD_GATEWAY_BASE", "", raising=False)
    monkeypatch.setattr(webapp_module.config, "ANALYTICS_GRAPHQL_URL", "", raising=False)
    monkeypatch.setattr(webapp_module.config, "AI_ASSISTANT_URL", "", raising=False)
    monkeypatch.setattr(webapp_module.config, "EXTERNAL_SERVICE_LINKS", "", raising=False)

    with app.test_client() as client:
        response = client.get("/", base_url="http://localhost:8000/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-api-base="http://localhost:8000"' in html
    assert 'data-analytics-graphql=""' in html
    assert 'data-ai-endpoint=""' in html


def test_index_respects_configured_gateway(monkeypatch):
    """Si se configura un gateway explícito, la plantilla debe usarlo."""

    monkeypatch.setattr(
        webapp_module.config,
        "DASHBOARD_GATEWAY_BASE",
        "http://localhost:8080",
        raising=False,
    )
    monkeypatch.setattr(
        webapp_module.config,
        "ANALYTICS_GRAPHQL_URL",
        "http://analytics.local/graphql",
        raising=False,
    )
    monkeypatch.setattr(
        webapp_module.config,
        "AI_ASSISTANT_URL",
        "http://assistant.local/chat",
        raising=False,
    )
    monkeypatch.setattr(webapp_module.config, "EXTERNAL_SERVICE_LINKS", "", raising=False)

    with app.test_client() as client:
        response = client.get("/", base_url="http://localhost:8000/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-api-base="http://localhost:8080"' in html
    assert 'data-analytics-graphql="http://analytics.local/graphql"' in html
    assert 'data-ai-endpoint="http://assistant.local/chat"' in html


@pytest.mark.integration
def test_dashboard_live_server_same_origin(tmp_path, monkeypatch):
    """El dashboard debe funcionar íntegramente contra el backend local."""

    open_trades.clear()
    closed_trades.clear()

    open_trades.append(
        {
            "symbol": "BTC_USDT",
            "entry_price": 100.0,
            "quantity": 2.0,
            "side": "BUY",
        }
    )

    history_file = tmp_path / "history.csv"
    history_file.write_text(
        "symbol,side,quantity,entry_price,exit_price,take_profit,stop_loss,profit,open_time,close_time\n"
        "BTC_USDT,BUY,2.0,100.0,105.0,0,0,10.0,2024-01-01T00:00:00,2024-01-01T02:00:00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp_module, "HISTORY_FILE", history_file)
    monkeypatch.setattr(webapp_module.config, "DASHBOARD_GATEWAY_BASE", "", raising=False)
    monkeypatch.setattr(webapp_module.config, "ANALYTICS_GRAPHQL_URL", "", raising=False)
    monkeypatch.setattr(webapp_module.config, "AI_ASSISTANT_URL", "", raising=False)
    monkeypatch.setattr(webapp_module.config, "EXTERNAL_SERVICE_LINKS", "", raising=False)

    original_testing = app.config.get("TESTING", False)
    app.config.update(TESTING=False)

    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    index_response = None

    try:
        base_url = f"http://127.0.0.1:{server.server_port}"

        # Espera breve para garantizar que el servidor esté listo.
        for _ in range(20):
            try:
                index_response = urllib_request.urlopen(base_url + "/", timeout=0.5)
                break
            except urllib_error.URLError:
                time.sleep(0.05)
        else:
            pytest.fail("El servidor Flask no se inicializó a tiempo")

        assert index_response is not None
        assert index_response.status == 200
        html = index_response.read().decode("utf-8")
        index_response.close()
        assert 'data-api-base="http://127.0.0.1:' in html
        assert 'data-analytics-graphql=""' in html
        assert 'data-ai-endpoint=""' in html

        with urllib_request.urlopen(base_url + "/api/trades", timeout=1) as trades_resp:
            assert trades_resp.status == 200
            trades_payload = json.load(trades_resp)
        assert trades_payload[0]["symbol"] == "BTC_USDT"

        with urllib_request.urlopen(base_url + "/api/summary", timeout=1) as summary_resp:
            assert summary_resp.status == 200
            summary_payload = json.load(summary_resp)
        assert summary_payload["total_positions"] == 1
        assert summary_payload["per_symbol"][0]["symbol"] == "BTC_USDT"
        assert summary_payload["trading_active"] is True

        with urllib_request.urlopen(base_url + "/api/history", timeout=1) as history_resp:
            assert history_resp.status == 200
            history_payload = json.load(history_resp)
        assert history_payload[0]["symbol"] == "BTC_USDT"
    finally:
        server.shutdown()
        thread.join(timeout=1)
        app.config.update(TESTING=original_testing)
        open_trades.clear()
        closed_trades.clear()
