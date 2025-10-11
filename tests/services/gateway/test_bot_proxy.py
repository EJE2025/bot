from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import pytest

from services.gateway.app import BOT_SERVICE_URL, app
TransportHandler = Callable[[httpx.Request], httpx.Response]


async def _call_gateway(
    method: str,
    path: str,
    handler: TransportHandler,
    *,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> httpx.Response:
    transport = httpx.MockTransport(handler)
    bot_client = httpx.AsyncClient(transport=transport, base_url=BOT_SERVICE_URL)
    previous_client = getattr(app.state, "bot_client", None)
    app.state.bot_client = bot_client

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as test_client:
            response = await test_client.request(method, path, headers=headers, **kwargs)
    finally:
        await bot_client.aclose()
        if previous_client is not None:
            app.state.bot_client = previous_client
        else:
            if hasattr(app.state, "bot_client"):
                delattr(app.state, "bot_client")

    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path,bot_path,request_kwargs,expected_response",
    [
        ("GET", "/api/trades", "/api/trades", {"params": {"status": "open"}}, {"trades": []}),
        ("GET", "/api/summary", "/api/summary", {}, {"summary": {"pnl": 12}}),
        ("GET", "/api/history", "/api/history", {}, {"history": ["trade"]}),
        (
            "POST",
            "/api/toggle-trading",
            "/api/toggle-trading",
            {"json": {"enabled": True}},
            {"status": "ok"},
        ),
        (
            "POST",
            "/api/trades/123/close",
            "/api/trades/123/close",
            {"json": {"reason": "manual"}},
            {"result": "closed"},
        ),
        (
            "POST",
            "/api/trades/123/close-partial",
            "/api/trades/123/close-partial",
            {"json": {"amount": 0.5}},
            {"result": "partial"},
        ),
    ],
)
async def test_gateway_forwards_success(
    method: str,
    path: str,
    bot_path: str,
    request_kwargs: Dict[str, Any],
    expected_response: Dict[str, Any],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == method
        assert request.url.path == bot_path
        if "params" in request_kwargs:
            assert request.url.params == httpx.QueryParams(request_kwargs["params"])
        if "json" in request_kwargs:
            assert request.headers.get("content-type") == "application/json"
            assert json.loads(request.content) == request_kwargs["json"]
        assert request.headers.get("x-extra-header") == "1"
        return httpx.Response(200, json=expected_response)

    response = await _call_gateway(
        method,
        path,
        handler,
        headers={"X-Extra-Header": "1"},
        **request_kwargs,
    )

    assert response.status_code == 200
    assert response.json() == expected_response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path,bot_path",
    [
        ("GET", "/api/trades", "/api/trades"),
        ("GET", "/api/summary", "/api/summary"),
        ("GET", "/api/history", "/api/history"),
        ("POST", "/api/toggle-trading", "/api/toggle-trading"),
        ("POST", "/api/trades/abc/close", "/api/trades/abc/close"),
        ("POST", "/api/trades/abc/close-partial", "/api/trades/abc/close-partial"),
    ],
)
async def test_gateway_propagates_errors(method: str, path: str, bot_path: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == bot_path
        return httpx.Response(503, content=b"bot down", headers={"content-type": "text/plain"})

    response = await _call_gateway(method, path, handler)

    assert response.status_code == 503
    assert response.text == "bot down"
