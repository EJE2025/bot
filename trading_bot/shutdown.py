"""Graceful shutdown helpers for the trading bot."""
from __future__ import annotations

import signal
import threading
from typing import Callable, List

_lock = threading.RLock()
_shutdown_requested = False
_callbacks: List[Callable[[], None]] = []
_STOP_EVENT = threading.Event()


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
    _STOP_EVENT.set()


def shutdown_requested() -> bool:
    with _lock:
        return _shutdown_requested


def get_stop_event() -> threading.Event:
    """Return the event toggled when a shutdown has been requested."""

    return _STOP_EVENT


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
    _STOP_EVENT.clear()
