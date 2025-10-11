"""API gateway aggregating multiple microservices."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Tuple

from starlette.datastructures import Headers

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

ANALYTICS_URL = os.getenv("ANALYTICS_URL", "http://analytics:5002/graphql")
AI_URL = os.getenv("AI_URL", "http://ai_service:5003")
TRADING_URL = os.getenv("TRADING_URL", "http://trading_engine:8000")
BOT_SERVICE_URL = os.getenv("BOT_SERVICE_URL", "http://trading_bot:8000")

app = FastAPI(title="Gateway")


@app.on_event("startup")
async def _create_bot_client() -> None:
    app.state.bot_client = httpx.AsyncClient(base_url=BOT_SERVICE_URL)


@app.on_event("shutdown")
async def _close_bot_client() -> None:
    bot_client: httpx.AsyncClient | None = getattr(app.state, "bot_client", None)
    if bot_client is not None:
        await bot_client.aclose()


_HOP_BY_HOP_HEADERS = {
    "host",
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
}


def _filter_hop_by_hop_header_items(
    header_items: Iterable[Tuple[str, str]]
) -> list[Tuple[str, str]]:
    return [
        (k, v)
        for k, v in header_items
        if k.lower() not in _HOP_BY_HOP_HEADERS
    ]


def _filtered_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    return dict(_filter_hop_by_hop_header_items(headers.items()))


async def _proxy_bot_request(request: Request, method: str, path: str) -> Response:
    body = await request.body()
    headers = _filtered_headers(request.headers)
    params = dict(request.query_params)

    bot_client: httpx.AsyncClient | None = getattr(app.state, "bot_client", None)
    if bot_client is None:
        raise HTTPException(status_code=503, detail="Bot service client not available")
    bot_response = await bot_client.request(
        method,
        path,
        content=body if body else None,
        params=params if params else None,
        headers=headers,
    )

    response_headers = Headers(
        raw=[
            (k.encode("latin-1"), v.encode("latin-1"))
            for k, v in _filter_hop_by_hop_header_items(bot_response.headers.multi_items())
        ]
    )
    return Response(
        content=bot_response.content,
        status_code=bot_response.status_code,
        headers=response_headers,
        media_type=bot_response.headers.get("content-type"),
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ai/report")
async def proxy_ai_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{AI_URL}/api/ai/report", json=payload, timeout=60)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.post("/ai/chat")
async def proxy_ai_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{AI_URL}/api/ai/chat", json=payload, timeout=60)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.get("/orders")
async def proxy_orders() -> Any:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{TRADING_URL}/orders", timeout=10)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.get("/api/trades")
async def proxy_bot_trades(request: Request) -> Response:
    return await _proxy_bot_request(request, "GET", "/api/trades")


@app.get("/api/summary")
async def proxy_bot_summary(request: Request) -> Response:
    return await _proxy_bot_request(request, "GET", "/api/summary")


@app.get("/api/history")
async def proxy_bot_history(request: Request) -> Response:
    return await _proxy_bot_request(request, "GET", "/api/history")


@app.post("/api/toggle-trading")
async def proxy_bot_toggle_trading(request: Request) -> Response:
    return await _proxy_bot_request(request, "POST", "/api/toggle-trading")


@app.post("/api/trades/{trade_id}/close")
async def proxy_bot_close_trade(request: Request, trade_id: str) -> Response:
    return await _proxy_bot_request(request, "POST", f"/api/trades/{trade_id}/close")


@app.post("/api/trades/{trade_id}/close-partial")
async def proxy_bot_close_partial_trade(request: Request, trade_id: str) -> Response:
    return await _proxy_bot_request(request, "POST", f"/api/trades/{trade_id}/close-partial")
