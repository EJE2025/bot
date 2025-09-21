"""Graceful shutdown helpers for the trading bot."""

from __future__ import annotations

import signal
import threading
from typing import Callable, List

_lock = threading.RLock()
_shutdown_requested = False
_callbacks: List[Callable[[], None]] = []


def _handler(signum, frame):  # pragma: no cover - signal module passes frame
    request_shutdown()


def install_signal_handlers() -> None:
    """Register SIGINT/SIGTERM handlers to request graceful shutdown."""

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def request_shutdown() -> None:
    """Mark shutdown as requested."""

    global _shutdown_requested
    with _lock:
        _shutdown_requested = True


def shutdown_requested() -> bool:
    with _lock:
        return _shutdown_requested


def register_callback(callback: Callable[[], None]) -> None:
    with _lock:
        _callbacks.append(callback)


def execute_callbacks() -> None:
    with _lock:
        callbacks = list(_callbacks)
        _callbacks.clear()
    for callback in callbacks:
        try:
            callback()
        except Exception:  # pragma: no cover - defensive logging handled upstream
            pass


def reset_for_tests() -> None:
    """Reset shutdown state (intended for unit tests)."""

    global _shutdown_requested
    with _lock:
        _shutdown_requested = False
        _callbacks.clear()
