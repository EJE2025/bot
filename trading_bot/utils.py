import time
from functools import wraps
from typing import Any, Callable


def normalize_symbol(sym: str) -> str:
    """Return a normalized symbol like ``BTC_USDT``
    regardless of separators."""
    if not sym:
        return ""
    s = sym.replace('/', '').replace('_', '').replace(':USDT', '').upper()
    if s.endswith('USDT'):
        return f"{s[:-4]}_USDT"
    return s


def circuit_breaker(
    max_failures: int = 3,
    reset_timeout: int = 60,
    fallback: Any = None,
):
    """Simple circuit breaker decorator.

    If ``max_failures`` consecutive calls raise an exception, the circuit opens
    and ``fallback`` is returned for subsequent calls until ``reset_timeout``
    seconds have passed.
    """

    def decorator(func: Callable):
        failures = 0
        opened_at: float | None = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failures, opened_at
            if opened_at is not None:
                if time.time() - opened_at < reset_timeout:
                    return fallback
                failures = 0
                opened_at = None
            try:
                result = func(*args, **kwargs)
                failures = 0
                return result
            except Exception:
                failures += 1
                if failures >= max_failures:
                    opened_at = time.time()
                raise

        return wrapper

    return decorator
