import logging
from typing import Dict, List
import requests
from . import config

logger = logging.getLogger(__name__)


def get_market_data(symbol: str, interval: str = "Min15", limit: int = 500) -> Dict:
    url = f"{config.BASE_URL_MEXC}/contract/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.error("MEXC error for %s: %s", symbol, data.get("message"))
            return {}
        return data.get("data", {})
    except Exception as exc:
        logger.error("Network error fetching %s: %s", symbol, exc)
        return {}


def get_ticker(symbol: str) -> Dict:
    """Return last price and bid/ask data for a MEXC futures symbol."""
    url = f"{config.BASE_URL_MEXC}/contract/ticker"
    params = {"symbol": symbol}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data.get("data", {})
        logger.error("Ticker error for %s: %s", symbol, data.get("message"))
    except Exception as exc:
        logger.error("Ticker network error for %s: %s", symbol, exc)
    return {}


def get_common_top_symbols(exchange, n: int = 15) -> List[str]:
    url = f"{config.BASE_URL_MEXC}/contract/detail"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Error fetching MEXC contracts: %s", exc)
        return []
    data = resp.json().get("data", [])
    mexc_list = []
    for item in data:
        if item.get("quoteCoin", "").upper() != "USDT":
            continue
        raw_sym = item.get("symbol", "")
        vol = item.get("volume24h") or item.get("turnover24h") or 0
        try:
            vol = float(vol)
        except Exception:
            vol = 0.0
        mexc_list.append((raw_sym, vol))
    mexc_list.sort(key=lambda x: x[1], reverse=True)
    mexc_symbols_sorted = [sym for sym, _ in mexc_list]

    def to_bitget_symbol(mexc_sym: str) -> str:
        return mexc_sym.replace("_", "/") + ":USDT"

    bitget_keys = set(exchange.markets.keys())
    common = [sym for sym in mexc_symbols_sorted if to_bitget_symbol(sym) in bitget_keys][:n]
    logger.info("Top %d common symbols: %s", len(common), common)
    return common


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
    """Fetch order book data from MEXC for a given symbol."""
    url = f"{config.BASE_URL_MEXC}/contract/depth"
    params = {"symbol": symbol, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.error("Order book error for %s: %s", symbol, data.get("message"))
            return {}
        return data.get("data", {})
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
