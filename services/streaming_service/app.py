"""Streaming service publishing market data over Redis Streams."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

from fastapi import FastAPI
from redis import asyncio as redis_async

STREAM_NAME = os.getenv("MARKET_STREAM", "market-data")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


app = FastAPI(title="Streaming Service")


@app.on_event("startup")
async def startup_event() -> None:
    app.state.redis = await redis_async.from_url(REDIS_URL)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    redis = getattr(app.state, "redis", None)
    if redis:
        await redis.close()


@app.post("/publish")
async def publish_event(event: Dict[str, Any]) -> Dict[str, Any]:
    redis = app.state.redis
    message_id = await redis.xadd(STREAM_NAME, {"payload": json.dumps(event)})
    return {"message_id": message_id}


async def publish_price(redis, symbol: str, price: float) -> str:
    return await redis.xadd(STREAM_NAME, {"payload": json.dumps({"symbol": symbol, "price": price})})


async def seed_sample_data(interval: float = 1.0) -> None:
    redis = await redis_async.from_url(REDIS_URL)
    try:
        price = 100.0
        while True:
            price += 0.5
            await publish_price(redis, "BTC-USD", round(price, 2))
            await asyncio.sleep(interval)
    finally:
        await redis.close()


if __name__ == "__main__":
    asyncio.run(seed_sample_data())
