import asyncio
import json
import logging
import threading
from collections import defaultdict

import websockets

# Order book depth to request
DEPTH = 20

# Public websocket URLs
MEXC_WS_URL = "wss://contract.mexc.com/ws"
BITGET_WS_URL = "wss://ws.bitget.com/mix/v1/stream"

# Symbol format mapping
MEXC_SYMBOL = lambda s: s.replace("/", "_").upper()
BITGET_SYMBOL = lambda s: s.replace("/", "-").upper()

# In-memory order book map
_liquidity = defaultdict(lambda: {"bids": {}, "asks": {}})

# Internal flag and loop
_loop = None
_thread = None

async def _mexc_listener(symbols):
    async with websockets.connect(MEXC_WS_URL, ping_interval=None) as ws:
        for sym in symbols:
            msg = {
                "method": "depth.subscribe",
                "params": [MEXC_SYMBOL(sym), DEPTH, "0"],
                "id": sym,
            }
            await ws.send(json.dumps(msg))
        async for message in ws:
            data = json.loads(message)
            if "data" not in data:
                continue
            sym = data.get("symbol") or data.get("channel", "").split("_")[0]
            if not sym:
                continue
            book = _liquidity[sym]
            book["bids"] = {float(p): float(v) for p, v in data["data"].get("bids", [])}
            book["asks"] = {float(p): float(v) for p, v in data["data"].get("asks", [])}

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
                sym = item.get("symbol") or item.get("instId")
                if not sym:
                    continue
                book = _liquidity[sym]
                book["bids"] = {float(b[0]): float(b[1]) for b in item.get("bids", [])}
                book["asks"] = {float(a[0]): float(a[1]) for a in item.get("asks", [])}

async def _run(symbols):
    await asyncio.gather(_mexc_listener(symbols), _bitget_listener(symbols))


def start(symbols):
    """Start websocket listeners in background thread."""
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
    if symbol:
        return _liquidity.get(symbol.upper())
    return dict(_liquidity)
