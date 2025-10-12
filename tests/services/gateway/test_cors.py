from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Dict

import httpx
import pytest
from fastapi import FastAPI


async def _options(
    app: FastAPI, url: str, *, origin: str, request_headers: Iterable[str]
) -> httpx.Response:
    headers: Dict[str, str] = {
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": ",".join(request_headers),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        return await client.options(url, headers=headers)


@pytest.mark.asyncio
async def test_cors_preflight_allows_configured_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway_module = importlib.import_module("services.gateway.app")

    with monkeypatch.context() as m:
        m.setenv("DASHBOARD_ALLOWED_ORIGINS", "https://dashboard.example.com")
        m.delenv("DASHBOARD_GATEWAY_BASE", raising=False)
        m.delenv("GATEWAY_BASE_URL", raising=False)
        gateway_module = importlib.reload(gateway_module)

        response = await _options(
            gateway_module.app,
            "/health",
            origin="https://dashboard.example.com",
            request_headers=["Content-Type"],
        )

        assert response.status_code == 200
        assert (
            response.headers.get("access-control-allow-origin")
            == "https://dashboard.example.com"
        )

        allow_methods = response.headers.get("access-control-allow-methods", "")
        assert "GET" in allow_methods
        assert "POST" in allow_methods

        allow_headers = {
            header.strip()
            for header in response.headers.get("access-control-allow-headers", "").split(",")
            if header
        }
        assert "Content-Type" in allow_headers or "*" in allow_headers

    importlib.reload(gateway_module)
