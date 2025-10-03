"""API gateway aggregating multiple microservices."""
from __future__ import annotations

import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException

ANALYTICS_URL = os.getenv("ANALYTICS_URL", "http://analytics:5002/graphql")
AI_URL = os.getenv("AI_URL", "http://ai_service:5003")
TRADING_URL = os.getenv("TRADING_URL", "http://trading_engine:8000")

app = FastAPI(title="Gateway")


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
