"""Runtime safety and permission helpers for trading operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import config

if False:  # pragma: no cover - typing helpers
    from ccxt import Exchange  # type: ignore

logger = logging.getLogger(__name__)


class PermissionError(RuntimeError):
    """Raised when a trading action is not permitted in the current context."""


def _is_paper_trading(exchange: Optional[object]) -> bool:
    """Return ``True`` when the current session is safe for paper trading."""
    if config.TEST_MODE:
        return True
    if config.TRADING_MODE != "live":
        return True
    if exchange is None:
        return False
    try:
        from .exchanges import MockExchange
    except Exception:  # pragma: no cover - fallback when optional deps missing
        return False
    return isinstance(exchange, MockExchange)


def _missing_credentials() -> list[str]:
    """Return a list of missing API credential names."""
    required: list[tuple[str, str]] = []
    if config.DEFAULT_EXCHANGE == "bitget":
        required.extend(
            [
                ("BITGET_API_KEY", config.BITGET_API_KEY),
                ("BITGET_API_SECRET", config.BITGET_API_SECRET),
                ("BITGET_PASSPHRASE", config.BITGET_PASSPHRASE),
            ]
        )
    elif config.DEFAULT_EXCHANGE == "binance":
        required.extend(
            [
                ("BINANCE_API_KEY", config.BINANCE_API_KEY),
                ("BINANCE_API_SECRET", config.BINANCE_API_SECRET),
            ]
        )
    elif config.DEFAULT_EXCHANGE == "mexc":
        required.extend(
            [
                ("MEXC_API_KEY", config.MEXC_API_KEY),
                ("MEXC_API_SECRET", config.MEXC_API_SECRET),
            ]
        )
    return [name for name, value in required if not value]


def _token_authorizes_live_trading() -> bool:
    """Validate the optional confirmation token file for live trading."""
    token_path = config.LIVE_TRADING_TOKEN_PATH
    if not token_path:
        return True
    path = Path(token_path).expanduser()
    if not path.exists():
        logger.error("Live trading token %s not found", path)
        return False
    try:
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        logger.error("Unable to stat %s: %s", path, exc)
        return False
    if mode & 0o077:
        logger.error("Token file %s must not be accessible to group/others", path)
        return False
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.error("Unable to read %s: %s", path, exc)
        return False
    expected = config.LIVE_TRADING_TOKEN_VALUE.strip()
    if expected and contents != expected:
        logger.error("Token file %s does not match expected confirmation", path)
        return False
    return True


def ensure_open_trade_allowed(exchange: Optional[object] = None) -> None:
    """Raise :class:`PermissionError` if opening positions is unsafe."""
    if _is_paper_trading(exchange):
        return
    if not config.ALLOW_LIVE_TRADING:
        raise PermissionError("Live trading disabled by configuration")
    if not _token_authorizes_live_trading():
        raise PermissionError("Live trading token missing or insecure")
    missing = _missing_credentials()
    if missing:
        raise PermissionError(f"Missing API credentials: {', '.join(missing)}")


def can_open_trade(exchange: Optional[object] = None) -> bool:
    """Return ``True`` if the bot is allowed to submit new orders."""
    try:
        ensure_open_trade_allowed(exchange)
    except PermissionError:
        return False
    return True


def audit_environment(exchange: Optional[object] = None) -> None:
    """Log the current trading permissions to help operators verify safety."""
    if _is_paper_trading(exchange):
        logger.info("Trading in paper/sandbox mode (%s)", config.TRADING_MODE)
        return
    if not config.ALLOW_LIVE_TRADING:
        logger.warning("Live trading requested but ALLOW_LIVE_TRADING=0")
        return
    missing = _missing_credentials()
    if missing:
        logger.error("Missing credentials prevent live trading: %s", ", ".join(missing))
        return
    if not _token_authorizes_live_trading():
        logger.error("Live trading token validation failed")
        return
    logger.info("Live trading enabled on %s with secure token", config.DEFAULT_EXCHANGE)


__all__ = [
    "PermissionError",
    "ensure_open_trade_allowed",
    "can_open_trade",
    "audit_environment",
]
