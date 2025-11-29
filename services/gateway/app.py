from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

BOT_SERVICE_URL = os.getenv("BOT_SERVICE_URL", "http://localhost:8000")
_GATEWAY_BASE = os.getenv("GATEWAY_BASE_URL") or os.getenv(
    "DASHBOARD_GATEWAY_BASE", "*"
)

app = FastAPI(title="Trading Bot Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_GATEWAY_BASE] if _GATEWAY_BASE else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _copy_headers(source: httpx.Response) -> Dict[str, str]:
    excluded = {"content-length", "transfer-encoding", "connection"}
    return {
        key: value
        for key, value in source.headers.items()
        if key.lower() not in excluded
    }


async def _get_bot_client() -> httpx.AsyncClient:
    client = getattr(app.state, "bot_client", None)
    if client is None:
        client = httpx.AsyncClient(base_url=BOT_SERVICE_URL)
        app.state.bot_client = client
    return client


async def _forward_request(
    request: Request, path: str, extra_headers: Optional[Dict[str, str]] = None
) -> Response:
    client = await _get_bot_client()
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    if extra_headers:
        headers.update(extra_headers)
    params = dict(request.query_params)

    try:
        upstream = await client.send(
            client.build_request(
                request.method,
                path,
                content=body,
                headers=headers,
                params=params,
            ),
            stream=True,
        )
    except httpx.HTTPError:
        return Response(status_code=502, content=b"gateway error")

    async def _stream_content(resp: httpx.Response) -> Iterable[bytes]:
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    response_headers = _copy_headers(upstream)
    content_type = upstream.headers.get("content-type")

    if content_type and content_type.startswith("text/event-stream"):
        return StreamingResponse(
            _stream_content(upstream),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=content_type,
        )

    content = await upstream.aread()
    await upstream.aclose()
    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=content_type,
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.api_route("/api/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_api(full_path: str, request: Request) -> Response:
    return await _forward_request(request, f"/api/{full_path}")


@app.get("/events")
async def proxy_events(request: Request) -> Response:
    extra_headers = {"accept": "text/event-stream"}
    return await _forward_request(request, "/events", extra_headers=extra_headers)
