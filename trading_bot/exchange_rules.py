"""Helpers to validate orders against exchange trading rules."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from . import config


class ValidationError(Exception):
    """Raised when an order violates exchange constraints."""


@dataclass
class SymbolRules:
    price_tick: float
    qty_step: float
    min_qty: float
    min_notional: float | None = None
    max_leverage: float | None = None


_CACHE: Dict[str, tuple[SymbolRules, float]] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_TTL = 300.0


def _fetch_rules_from_exchange(symbol: str):
    try:
        from . import execution  # Local import to avoid circular dependency

        exchange = execution.exchange
    except Exception:
        exchange = None
    if exchange is None:
        return None
    market_symbol = symbol.replace("_", "/") + ":USDT"
    market = exchange.markets.get(market_symbol) if hasattr(exchange, "markets") else None
    if not market:
        return None
    precision = market.get("precision", {})
    limits = market.get("limits", {})
    price_tick = float(precision.get("price", 0.0) or market.get("tickSize", 0.0) or 0.01)
    qty_step = float(precision.get("amount", 0.0) or market.get("stepSize", 0.0) or 0.001)
    min_qty = float(limits.get("amount", {}).get("min", 0.0) or market.get("minQty", 0.0) or qty_step)
    min_notional = limits.get("cost", {}).get("min") or market.get("minNotional")
    max_leverage = market.get("maxLeverage")
    return SymbolRules(price_tick=price_tick, qty_step=qty_step, min_qty=min_qty, min_notional=min_notional, max_leverage=max_leverage)


def get_symbol_rules(symbol: str, *, fallback: Optional[SymbolRules] = None) -> SymbolRules:
    """Return :class:`SymbolRules` for ``symbol`` caching results for ``CACHE_TTL`` seconds."""

    now = time.time()
    with _CACHE_LOCK:
        if symbol in _CACHE:
            rules, ts = _CACHE[symbol]
            if now - ts <= _CACHE_TTL:
                return rules
        rules = _fetch_rules_from_exchange(symbol) or fallback or SymbolRules(0.01, 0.001, 0.001)
        _CACHE[symbol] = (rules, now)
        return rules


def quantize_price(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return math.floor(price / tick) * tick


def quantize_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def min_notional_usdt(symbol: str, rules: SymbolRules) -> float:
    """Return the minimum notional enforced for ``symbol`` in USDT."""

    if os.getenv("TEST_MODE") == "1" or getattr(config, "TEST_MODE", False):
        return float(rules.min_notional or 0.0)
    if not config.ENFORCE_EXCHANGE_MIN_NOTIONAL:
        return 0.0
    return float(rules.min_notional or 0.0)


def validate_order(symbol: str, side: str, price: float | None, qty: float, rules: SymbolRules) -> None:
    if qty <= 0:
        raise ValidationError("quantity must be positive")
    if rules.min_qty and qty < rules.min_qty:
        raise ValidationError(f"quantity {qty} below minimum {rules.min_qty}")
    if price is not None and price <= 0:
        raise ValidationError("price must be positive")
    min_notional = min_notional_usdt(symbol, rules)
    if min_notional and price is not None:
        notional = price * qty
        if notional < min_notional:
            raise ValidationError(
                f"notional {notional:.8f} below minimum {min_notional}"
            )


__all__ = [
    "ValidationError",
    "SymbolRules",
    "get_symbol_rules",
    "quantize_price",
    "quantize_qty",
    "min_notional_usdt",
    "validate_order",
]
