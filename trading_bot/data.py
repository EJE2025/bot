import logging
import time
from typing import Dict, Iterable, List
import os
import json
import requests
import pandas as pd
import numpy as np
from . import config
from .execution import exchange, MockExchange
from .utils import circuit_breaker

# How many attempts should be made for network calls
MAX_ATTEMPTS = config.DATA_RETRY_ATTEMPTS

logger = logging.getLogger(__name__)

CACHE_DIR = "cache"
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data_samples")


def normalize_symbol(sym: str) -> str:
    """Return compact symbol like ``BTCUSDT`` ignoring separators."""
    return sym.replace("/", "").replace("_", "").replace(":USDT", "").replace("-", "").upper()


def display_symbol(sym: str) -> str:
    """Return symbol formatted with underscore as ``BTC_USDT``."""
    norm = normalize_symbol(sym)
    if norm.endswith("USDT") and len(norm) > 4:
        return norm[:-4] + "_USDT"
    return norm


def _generic_cache_path(prefix: str, *parts: str) -> str:
    name = "_".join((prefix, *parts)) + ".json"
    return os.path.join(CACHE_DIR, name)

def _cache_path(symbol: str, interval: str, limit: int) -> str:
    name = f"{symbol}_{interval}_{limit}.json"
    return os.path.join(CACHE_DIR, name)


def _to_binance_interval(interval: str) -> str:
    if interval.startswith("Min"):
        mins = interval[3:]
        return f"{mins}m"
    return interval


@circuit_breaker(fallback=None, max_failures=3, reset_timeout=30)
def get_market_data(symbol: str, interval: str = "Min15", limit: int = 500) -> Dict | None:
    """Return OHLCV data from Binance futures.

    If the request fails ``MAX_ATTEMPTS`` times it falls back to cached data
    stored under ``cache/``. ``None`` is returned when no cache is available.
    """
    symbol_raw = symbol.replace("_", "")

    # Look for local CSV samples before making any network calls
    csv_file = os.path.join(SAMPLE_DATA_DIR, f"{display_symbol(symbol)}_{interval}.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = df.head(limit)
        return {
            "close": df["close"].tolist(),
            "high": df["high"].tolist(),
            "low": df["low"].tolist(),
            "vol": df["vol"].tolist(),
        }

    if config.TEST_MODE:
        rng = np.random.default_rng(hash(symbol) % 2**32)
        base_price = 100.0
        price = base_price + rng.standard_normal(limit).cumsum() * 0.5
        high = price + rng.uniform(0.1, 1.0, size=limit)
        low = price - rng.uniform(0.1, 1.0, size=limit)
        vol = rng.uniform(100, 1000, size=limit)
        return {
            "close": price.tolist(),
            "high": high.tolist(),
            "low": low.tolist(),
            "vol": vol.tolist(),
        }

    url = f"{config.BASE_URL_BINANCE}/fapi/v1/klines"
    params = {"symbol": symbol_raw, "interval": _to_binance_interval(interval), "limit": limit}
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError("API error")
            parsed = {
                "close": [x[4] for x in data],
                "high": [x[2] for x in data],
                "low": [x[3] for x in data],
                "vol": [x[5] for x in data],
            }
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(_cache_path(symbol_raw, interval, limit), "w", encoding="utf-8") as fh:
                json.dump(parsed, fh)
            return parsed
        except Exception as exc:
            logger.warning(
                "Network error fetching %s (attempt %d/%d): %s",
                symbol,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
            )
            time.sleep(2 ** attempt)

    path = _cache_path(symbol_raw, interval, limit)
    if os.path.exists(path):
        logger.info("Using cached data for %s", symbol)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def get_ticker(symbol: str) -> Dict | None:
    """Return last price and bid/ask data for a Binance futures symbol.

    Falls back to cached data when the request repeatedly fails. ``None`` is
    returned if no cache exists.
    """
    url = f"{config.BASE_URL_BINANCE}/fapi/v1/ticker/bookTicker"
    params = {"symbol": symbol.replace("_", "")}
    cache_file = _generic_cache_path("ticker", symbol)
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            return data
        except Exception as exc:
            logger.warning(
                "Ticker network error for %s attempt %d/%d: %s",
                symbol,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
            )
            time.sleep(2 ** attempt)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as fh:
            logger.info("Using cached ticker for %s", symbol)
            return json.load(fh)
    return None


def get_common_top_symbols(exchange, n: int = 15,
                           exclude: Iterable[str] | None = None) -> List[str]:
    """Return the most liquid symbols according to ``exchange.fetch_markets``.

    If the request fails multiple times, the cached ``exchange.markets`` mapping
    is used as a fallback.
    """
    markets = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            markets = exchange.fetch_markets()
            break
        except Exception as exc:
            logger.warning(
                "Error fetching markets attempt %d/%d: %s",
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
            )
            time.sleep(2 ** attempt)
    if markets is None:
        markets = getattr(exchange, "markets", {})

    if isinstance(markets, list):
        market_dict = {m.get("symbol"): m for m in markets if isinstance(m, dict)}
    else:
        market_dict = markets or {}

    # symbols to exclude normalized
    excluded = {
        normalize_symbol(s)
        for s in (config.BLACKLIST_SYMBOLS | config.UNSUPPORTED_SYMBOLS)
    }
    if exclude:
        excluded.update(normalize_symbol(s) for s in exclude)

    # Keep only USDT pairs and not excluded
    filtered = [
        m for m in market_dict.values()
        if m.get("symbol", "").upper().endswith("USDT")
        and normalize_symbol(m.get("symbol", "")) not in excluded
    ]
    sorted_by_vol = sorted(
        filtered,
        key=lambda m: float(m.get("volume") or 0),
        reverse=True,
    )

    top = [display_symbol(m["symbol"]) for m in sorted_by_vol[:n]]
    logger.info("Top %d symbols by volume: %s", len(top), top)
    return top


def get_current_price_ticker(symbol: str) -> float | None:
    """Return the latest traded price for the given symbol.

    In test mode or when using :class:`MockExchange`, prices are obtained from
    the mock directly to avoid inconsistencies with real network data.
    Returns ``None`` when the request cannot be completed and no cached value is
    available.
    """
    sym_ccxt = symbol.replace("_", "/") + ":USDT"
    if config.TEST_MODE or isinstance(exchange, MockExchange):
        try:
            return exchange._get_market_price(sym_ccxt)
        except Exception as exc:  # pragma: no cover - mock rarely fails
            logger.error("Mock price retrieval failed for %s: %s", symbol, exc)
            return None

    bitget_sym = symbol.replace("_", "") + "_UMCBL"
    endpoint = "/api/mix/v1/market/ticker"
    params = {"symbol": bitget_sym, "productType": "USDT-FUTURES"}
    url = config.BASE_URL_BITGET + endpoint
    cache_file = _generic_cache_path("price", symbol)
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "00000" and "data" in data:
                price = float(data["data"]["last"])
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as fh:
                    json.dump({"price": price}, fh)
                return price
        except Exception as exc:
            logger.warning(
                "Ticker error for %s attempt %d/%d: %s",
                symbol,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
            )
            time.sleep(2 ** attempt)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as fh:
            logger.info("Using cached price for %s", symbol)
            return json.load(fh).get("price")
    return None


def get_order_book(symbol: str, limit: int = 50) -> Dict | None:

    """Fetch order book data from Binance for a given symbol.

    On repeated failures the cached order book is returned when available.
    """
    url = f"{config.BASE_URL_BINANCE}/fapi/v1/depth"
    params = {"symbol": symbol.replace("_", ""), "limit": limit}
    cache_file = _generic_cache_path("book", f"{symbol}_{limit}")
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 400:
                logger.error(
                    "Símbolo no válido para order book %s: %s",
                    symbol,
                    resp.text,
                )
                return None
            resp.raise_for_status()
            data = resp.json()
            book = {"bids": data.get("bids", []), "asks": data.get("asks", [])}
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(book, fh)
            return book
        except Exception as exc:
            logger.warning(
                "Order book network error for %s attempt %d/%d: %s",
                symbol,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
            )
            time.sleep(2 ** attempt)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as fh:
            logger.info("Using cached order book for %s", symbol)
            return json.load(fh)
    return None


def order_book_imbalance(book: Dict, price: float, pct: float = 0.01) -> float:
    """Return the bid minus ask volume near the current price."""
    if not book:
        return 0.0
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    bid_vol = sum(float(q) for p, q in bids if float(p) >= price * (1 - pct))
    ask_vol = sum(float(q) for p, q in asks if float(p) <= price * (1 + pct))
    return bid_vol - ask_vol


def top_liquidity_levels(book: Dict | None, n: int = 5):
    """Return the highest volume bid and ask levels."""
    if not book:
        return [], []
    bids = [(float(p), float(q)) for p, q in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])]
    bids.sort(key=lambda x: x[1], reverse=True)
    asks.sort(key=lambda x: x[1], reverse=True)
    return bids[:n], asks[:n]
