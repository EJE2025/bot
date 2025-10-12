"""API gateway aggregating multiple microservices."""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ANALYTICS_URL = os.getenv("ANALYTICS_URL", "http://analytics:5002/graphql")
AI_URL = os.getenv("AI_URL", "http://ai_service:5003")
TRADING_URL = os.getenv("TRADING_URL", "http://trading_engine:8000")
BOT_SERVICE_URL = os.getenv("BOT_SERVICE_URL", "http://trading_bot:8000")

app = FastAPI(title="Gateway")


def _parse_origins(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def _allowed_origins() -> Sequence[str]:
    configured_origins = _parse_origins(os.getenv("DASHBOARD_ALLOWED_ORIGINS"))
    if not configured_origins:
        configured_origins = _parse_origins(
            os.getenv("DASHBOARD_GATEWAY_BASE") or os.getenv("GATEWAY_BASE_URL")
        )

    return configured_origins or ["*"]


allowed_origins = list(_allowed_origins())

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.on_event("startup")
async def _create_bot_client() -> None:
    app.state.bot_client = httpx.AsyncClient(base_url=BOT_SERVICE_URL)


@app.on_event("shutdown")
async def _close_bot_client() -> None:
    bot_client: httpx.AsyncClient | None = getattr(app.state, "bot_client", None)
    if bot_client is not None:
        await bot_client.aclose()


def _filtered_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    excluded = {"host", "content-length", "content-encoding", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in excluded}


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

    response_headers = _filtered_headers(bot_response.headers)
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
