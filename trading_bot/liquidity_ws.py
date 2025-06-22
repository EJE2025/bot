

from __future__ import annotations


import asyncio
import json
import logging
import threading
from collections import defaultdict

from typing import Iterable

import websockets


logger = logging.getLogger(__name__)

# Order book depth to request
DEPTH = 20

# Public websocket URLs
BITGET_WS_URL = "wss://ws.bitget.com/mix/v1/stream"

BINANCE_WS_BASE = "wss://fstream.binance.com/stream?streams="

# Symbol format mapping
BINANCE_SYMBOL = lambda s: s.replace("/", "").replace("_", "").upper()
BITGET_SYMBOL = lambda s: s.replace("/", "-").upper()

# In-memory order book map
_liquidity: dict[str, dict[str, dict[float, float]]] = defaultdict(
    lambda: {"bids": {}, "asks": {}}
)

# Internal flag and loop
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()



async def _binance_listener(symbols: Iterable[str]):
    streams = "/".join([f"{BINANCE_SYMBOL(sym).lower()}@depth{DEPTH}@100ms" for sym in symbols])
    url = BINANCE_WS_BASE + streams
    async with websockets.connect(url, ping_interval=None) as ws:
        async for message in ws:
            data = json.loads(message)
            content = data.get("data")
            if not content or "bids" not in content:
                continue
            sym_raw = content.get("s")
            if not sym_raw:
                continue
            sym = sym_raw[:-4] + "_" + sym_raw[-4:]
            with _lock:
                book = _liquidity[sym]
                book["bids"] = {float(p): float(q) for p, q in content.get("bids", [])}
                book["asks"] = {float(p): float(q) for p, q in content.get("asks", [])}

async def _bitget_listener(symbols):
    subs = [
        {
            "op": "subscribe",
            "args": [f"books5:{BITGET_SYMBOL(sym)}"]
        }
        for sym in symbols
    ]
    async with websockets.connect(BITGET_WS_URL, ping_interval=None) as ws:
        for sub in subs:
            await ws.send(json.dumps(sub))
        async for message in ws:
            data = json.loads(message)
            if "data" not in data:
                continue
            for item in data["data"]:
                if not isinstance(item, dict):
                    logger.warning("Unexpected Bitget payload: %s", data)
                    continue
                sym = item.get("symbol") or item.get("instId")
                if not sym:
                    continue

                with _lock:
                    book = _liquidity[sym]
                    book["bids"] = {float(b[0]): float(b[1]) for b in item.get("bids", [])}
                    book["asks"] = {float(a[0]): float(a[1]) for a in item.get("asks", [])}

async def _run(symbols):
    await asyncio.gather(_binance_listener(symbols), _bitget_listener(symbols))


def start(symbols):
    """Start or update websocket listeners in a background thread."""
    global _loop, _thread
    if _loop:
        return
    _loop = asyncio.new_event_loop()

    def runner():
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(_run(symbols))


    _thread = threading.Thread(target=runner, daemon=True)
    _thread.start()


def get_liquidity(symbol=None):
    """Get current liquidity map. If symbol provided, return its book."""

    with _lock:
        if symbol:
            return _liquidity.get(symbol.upper())
        return {k: v.copy() for k, v in _liquidity.items()}


# Example usage to start streaming the top 15 symbols:
# top_symbols = ["BTC_USDT", "ETH_USDT", ...]  # output from data.get_common_top_symbols()
# start(top_symbols)

