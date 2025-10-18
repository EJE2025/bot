from __future__ import annotations

import ipaddress
import threading
import time
from typing import Any

import requests

try:
    import webview
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyWebview es necesario para el modo de escritorio. Instala 'pywebview'."
    ) from exc

from . import config, webapp

_WINDOW: webview.Window | None = None


def _start_dashboard() -> None:
    """Run the Flask dashboard inside the current thread."""

    webapp.start_dashboard(config.WEBAPP_HOST, config.WEBAPP_PORT)


def _client_host() -> str:
    """Return the host that desktop clients should use to reach the dashboard."""

    host = config.WEBAPP_HOST.strip() or "127.0.0.1"
    if host in {"*", "0", "[::]"}:
        return "127.0.0.1"

    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return host

    if parsed.is_unspecified:
        return "127.0.0.1" if parsed.version == 4 else "::1"
    return host


def _base_url() -> str:
    host = _client_host()
    needs_brackets = ":" in host and not host.startswith("[")
    client_host = f"[{host}]" if needs_brackets else host
    return f"http://{client_host}:{config.WEBAPP_PORT}"


def _wait_health(timeout: float = 30.0) -> bool:
    """Wait until the dashboard health endpoint reports success."""

    base = _base_url()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base}/api/health", timeout=1.5)
            if response.ok:
                payload: dict[str, Any] = response.json()  # type: ignore[assignment]
                if payload.get("ok"):
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def launch_desktop() -> None:
    """Start the dashboard server and display it inside a native window."""

    server = threading.Thread(target=_start_dashboard, daemon=True)
    server.start()

    ok = _wait_health(30.0)
    base = _base_url()
    if not ok:
        # Proceed anyway to allow inspecting potential startup errors.
        pass

    global _WINDOW
    _WINDOW = webview.create_window(
        "Trading Bot Dashboard", base, width=1280, height=820, resizable=True
    )
    webview.start()


def eval_js(script: str) -> None:
    """Evaluate JavaScript code inside the desktop window if available."""

    if not _WINDOW:
        return
    try:
        _WINDOW.evaluate_js(script)
    except Exception:
        # Evaluation errors should not crash the trading runtime.
        pass
