"""Utilities to avoid duplicated order submissions."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _CacheEntry:
    created_at: float
    payload: Any


class IdempotencyCache:
    """Simple in-memory cache with TTL semantics for idempotency keys."""

    def __init__(self, ttl_seconds: float = 60.0, max_size: int = 512) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._items: Dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [key for key, entry in self._items.items() if now - entry.created_at > self.ttl_seconds]
        for key in expired:
            self._items.pop(key, None)
        # Keep dictionary bounded even if TTL not triggered yet
        if len(self._items) > self.max_size:
            for key in list(self._items.keys())[: len(self._items) - self.max_size]:
                self._items.pop(key, None)

    def add(self, key: str, payload: Any) -> None:
        with self._lock:
            self._purge_expired()
            self._items[key] = _CacheEntry(time.time(), payload)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._purge_expired()
            entry = self._items.get(key)
            return entry.payload if entry is not None else None

    def reserve(self, key: str) -> bool:
        """Reserve ``key`` returning ``False`` if it already exists."""

        with self._lock:
            self._purge_expired()
            if key in self._items:
                return False
            self._items[key] = _CacheEntry(time.time(), None)
            return True


_CACHE = IdempotencyCache()


def build_idempotency_key(
    symbol: str,
    side: str,
    price: float,
    quantity: float,
    bucket_seconds: int = 60,
) -> str:
    """Return a deterministic key for a trade request bucketed by ``bucket_seconds``."""

    bucket = int(time.time() // bucket_seconds)
    raw = f"{symbol}|{side}|{round(price, 8)}|{round(quantity, 8)}|{bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def should_submit(key: str) -> bool:
    """Return ``True`` if ``key`` was not seen recently."""

    return _CACHE.reserve(key)


def store_result(key: str, payload: Any) -> None:
    """Store ``payload`` associated with ``key`` for future retrieval."""

    _CACHE.add(key, payload)


def get_cached_result(key: str) -> Optional[Any]:
    """Return cached result for ``key`` if available."""

    return _CACHE.get(key)


def reset_cache(ttl_seconds: float | None = None) -> None:
    """Reset the in-memory cache, optionally overriding the TTL."""

    global _CACHE
    ttl = ttl_seconds if ttl_seconds is not None else _CACHE.ttl_seconds
    _CACHE = IdempotencyCache(ttl_seconds=ttl, max_size=_CACHE.max_size)
