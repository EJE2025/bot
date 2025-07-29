from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    """Return a normalized symbol without separators or suffixes."""
    return symbol.replace("/", "").replace("_", "").replace(":USDT", "").upper()

