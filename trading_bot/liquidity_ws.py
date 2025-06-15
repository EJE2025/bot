"""Order book streaming helpers for liquidity heatmap."""

import asyncio
import json
import logging
import threading
from collections import defaultdict

import websockets
from unicorn_mexc_websocket_api.manager import MexcWebsocketApiManager

logger = logging.getLogger(__name__)

# Order book depth to request
DEPTH = 20

# Public websocket URLs
BITGET_WS_URL = "wss://ws.bitget.com/mix/v1/stream"

# Symbol format mapping
MEXC_SYMBOL = lambda s: s.replace("/", "_").upper()
BITGET_SYMBOL = lambda s: s.replace("/", "-").upper()

# In-memory order book map
_liquidity = defaultdict(lambda: {"bids": {}, "asks": {}})

# Internal flag and loop
_loop = None
_thread = None
_mexc_manager = None


def _mexc_callback(message):
    """Process depth updates from MEXC."""
    data = message.get("data")
    if not isinstance(data, dict):
        return
    sym = message.get("symbol") or data.get("symbol")
    if not sym:
        return
    book = _liquidity[sym]
    book["bids"] = {float(p): float(v) for p, v in data.get("bids", [])}
    book["asks"] = {float(p): float(v) for p, v in data.get("asks", [])}


def _mexc_listener(symbols):
    global _mexc_manager
    if _mexc_manager is not None:
        return
    _mexc_manager = MexcWebsocketApiManager()
    subs = [MEXC_SYMBOL(s) for s in symbols]
    _mexc_manager.subscribe_depth_stream(subs, depth=DEPTH, callback=_mexc_callback)

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
                book = _liquidity[sym]
                book["bids"] = {float(b[0]): float(b[1]) for b in item.get("bids", [])}
                book["asks"] = {float(a[0]): float(a[1]) for a in item.get("asks", [])}

async def _run(symbols):
    await _bitget_listener(symbols)


def start(symbols):
    """Start websocket listeners in background thread."""
    global _loop, _thread
    if _loop:
        return
    _mexc_listener(symbols)
    _loop = asyncio.new_event_loop()
    def runner():
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(_run(symbols))
    _thread = threading.Thread(target=runner, daemon=True)
    _thread.start()


def get_liquidity(symbol=None):
    """Get current liquidity map. If symbol provided, return its book."""
    if symbol:
        return _liquidity.get(symbol.upper())
    return dict(_liquidity)


# Example usage:
# start(["BTC/USDT", "ETH/USDT"])
