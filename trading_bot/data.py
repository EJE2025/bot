import logging
import time
from typing import Dict, List
import os
import json
import requests
from . import config

logger = logging.getLogger(__name__)

CACHE_DIR = "cache"

def _cache_path(symbol: str, interval: str, limit: int) -> str:
    name = f"{symbol}_{interval}_{limit}.json"
    return os.path.join(CACHE_DIR, name)


def _to_binance_interval(interval: str) -> str:
    if interval.startswith("Min"):
        mins = interval[3:]
        return f"{mins}m"
    return interval


def get_market_data(symbol: str, interval: str = "Min15", limit: int = 500) -> Dict:
    symbol_raw = symbol.replace("_", "")
    url = f"{config.BASE_URL_BINANCE}/fapi/v1/klines"
    params = {"symbol": symbol_raw, "interval": _to_binance_interval(interval), "limit": limit}
    for attempt in range(3):
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
            logger.warning("Network error fetching %s (attempt %d): %s", symbol, attempt + 1, exc)
            time.sleep(2 ** attempt)
    # fallback to cache

    path = _cache_path(symbol_raw, interval, limit)

    if os.path.exists(path):
        logger.info("Using cached data for %s", symbol)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def get_ticker(symbol: str) -> Dict:
    """Return last price and bid/ask data for a Binance futures symbol."""
    url = f"{config.BASE_URL_BINANCE}/fapi/v1/ticker/bookTicker"
    params = {"symbol": symbol.replace("_", "")}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return data
    except Exception as exc:
        logger.error("Ticker network error for %s: %s", symbol, exc)
    return {}


def get_common_top_symbols(exchange, n: int = 15) -> List[str]:
    """Return the most liquid symbols according to ``exchange.fetch_markets``."""
    try:
        markets = exchange.fetch_markets()
    except Exception as exc:
        logger.error("Error fetching markets: %s", exc)
        markets = getattr(exchange, "markets", {})

    if isinstance(markets, list):
        market_dict = {m.get("symbol"): m for m in markets if isinstance(m, dict)}
    else:
        market_dict = markets or {}

    # Keep only USDT pairs
    filtered = [
        m for m in market_dict.values()
        if m.get("symbol", "").upper().endswith("USDT")
    ]
    sorted_by_vol = sorted(
        filtered,
        key=lambda m: float(m.get("volume") or 0),
        reverse=True,
    )

    def normalize(sym: str) -> str:
        return sym.replace("/", "_").replace(":USDT", "")

    top = [normalize(m["symbol"]) for m in sorted_by_vol[:n]]
    logger.info("Top %d symbols by volume: %s", len(top), top)
    return top


def get_current_price_ticker(symbol: str) -> float:
    """Return the latest traded price from Bitget for the given symbol."""
    bitget_sym = symbol.replace("_", "") + "_UMCBL"
    endpoint = "/api/mix/v1/market/ticker"
    params = {"symbol": bitget_sym, "productType": "USDT-FUTURES"}
    url = config.BASE_URL_BITGET + endpoint
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "00000" and "data" in data:
            return float(data["data"]["last"])
    except Exception as exc:
        logger.error("Ticker error for %s: %s", symbol, exc)
    return 0.0


def get_order_book(symbol: str, limit: int = 50) -> Dict:

    """Fetch order book data from Binance for a given symbol."""
    url = f"{config.BASE_URL_BINANCE}/fapi/v1/depth"
    params = {"symbol": symbol.replace("_", ""), "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {"bids": data.get("bids", []), "asks": data.get("asks", [])}
    except Exception as exc:
        logger.error("Order book network error for %s: %s", symbol, exc)
        return {}


def order_book_imbalance(book: Dict, price: float, pct: float = 0.01) -> float:
    """Return the bid minus ask volume near the current price."""
    if not book:
        return 0.0
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    bid_vol = sum(float(q) for p, q in bids if float(p) >= price * (1 - pct))
    ask_vol = sum(float(q) for p, q in asks if float(p) <= price * (1 + pct))
    return bid_vol - ask_vol


def top_liquidity_levels(book: Dict, n: int = 5):
    """Return the highest volume bid and ask levels."""
    bids = [(float(p), float(q)) for p, q in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])]
    bids.sort(key=lambda x: x[1], reverse=True)
    asks.sort(key=lambda x: x[1], reverse=True)
    return bids[:n], asks[:n]
