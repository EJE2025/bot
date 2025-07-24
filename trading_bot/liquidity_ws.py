from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from typing import Iterable
from . import exchanges, config, data

import websockets


logger = logging.getLogger(__name__)

# Fallback list if API call fails
DEFAULT_TOP_15_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
    "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LTCUSDT",
    "BCHUSDT", "LINKUSDT", "UNIUSDT", "ALGOUSDT", "ATOMUSDT",
]


def get_top_15_symbols() -> list[str]:
    """Return the 15 highest volume USDT futures symbols."""
    ex = exchanges.get_exchange(config.DEFAULT_EXCHANGE)
    symbols = data.get_common_top_symbols(ex, 15)
    if not symbols:
        return DEFAULT_TOP_15_SYMBOLS
    return [s.replace("_", "") for s in symbols]

# Order book depth to request and stream interval
DEPTH = 10
BINANCE_INTERVAL_MS = 500

# Default symbols analysed by the bot
TOP_15_SYMBOLS = get_top_15_symbols()

# Public websocket URLs
BITGET_WS_URL = "wss://ws.bitget.com/mix/v1/stream"
BINANCE_WS_BASE = "wss://fstream.binance.com/stream?streams="

# Symbol format mapping
BINANCE_SYMBOL = lambda s: s.replace("/", "").replace("_", "").upper()
BITGET_SYMBOL = lambda s: s.replace("/", "-").upper()


def format_binance_symbol(sym_raw: str) -> str:
    """Return symbol formatted as ``BTC_USDT (Binance)``."""
    base = sym_raw[:-4]
    quote = sym_raw[-4:]
    return f"{base}_{quote} (Binance)"


def format_binance_stream_symbol(sym_raw: str) -> str:
    """Return stream part for Binance, e.g. btcusdt@depth10@500ms."""
    return BINANCE_SYMBOL(sym_raw).lower() + f"@depth{DEPTH}@{BINANCE_INTERVAL_MS}ms"


def format_bitget_symbol(sym: str) -> str:
    """Normalize Bitget symbol to ``BTC_USDT (Bitget)`` format."""
    if sym.endswith("_UMCBL"):
        base_quote = sym.replace("_UMCBL", "")
        base = base_quote[:-4]
        quote = base_quote[-4:]
        return f"{base}_{quote} (Bitget)"
    if "-" in sym:
        base, quote = sym.split("-")
        return f"{base}_{quote} (Bitget)"
    base = sym[:-4]
    quote = sym[-4:]
    return f"{base}_{quote} (Bitget)"


def format_bitget_stream_symbol(sym: str) -> str:
    """Convert symbol to Bitget WS format like BTC-USDT."""
    cleaned = sym.replace("/", "").replace("_", "").upper()
    if cleaned.endswith("USDT") and len(cleaned) > 4:
        return f"{cleaned[:-4]}-USDT"
    return cleaned

# In-memory order book map
_liquidity: dict[str, dict[str, dict[float, float]]] = defaultdict(
    lambda: {"bids": {}, "asks": {}}
)

# Internal flag and loop
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()

# Running websocket connections
_ws_binance = None
_ws_bitget = None

# Stop flag for graceful shutdown
_stop_event = threading.Event()

# Reconnection delay in seconds
RECONNECT_DELAY = 5



async def _binance_listener(symbols: Iterable[str]):
    streams = "/".join(
        [format_binance_stream_symbol(sym) for sym in symbols]
    )
    url = BINANCE_WS_BASE + streams
    global _ws_binance
    while not _stop_event.is_set():
        try:
            logger.info("Connecting to Binance WS: %s", url)
            async with websockets.connect(url, ping_interval=None) as ws:
                _ws_binance = ws
                logger.info("Binance WS connected")
                async for message in ws:
                    if _stop_event.is_set():
                        break
                    data = json.loads(message)
                    content = data.get("data")
                    if not content or "bids" not in content:
                        continue
                    sym_raw = content.get("s")
                    if not sym_raw:
                        continue
                    sym = format_binance_symbol(sym_raw)
                    with _lock:
                        book = _liquidity[sym]
                        book["bids"] = {
                            float(p): float(q) for p, q in content.get("bids", [])
                        }
                        book["asks"] = {
                            float(p): float(q) for p, q in content.get("asks", [])
                        }
        except (websockets.ConnectionClosed, asyncio.TimeoutError) as exc:
            if _stop_event.is_set():
                break
            logger.warning(
                "Binance WS disconnected: %s. Reconnecting in %ss...",
                exc,
                RECONNECT_DELAY,
            )
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as exc:
            logger.error("Unexpected error in Binance listener: %s", exc, exc_info=True)
            if not _stop_event.is_set():
                await asyncio.sleep(RECONNECT_DELAY)
    _ws_binance = None

async def _bitget_listener(symbols):
    subs = [
        {
            "op": "subscribe",
            "args": [f"books{DEPTH}:{format_bitget_stream_symbol(sym)}"]
        }
        for sym in symbols
    ]
    global _ws_bitget
    while not _stop_event.is_set():
        try:
            logger.info("Connecting to Bitget WS: %s", BITGET_WS_URL)
            async with websockets.connect(BITGET_WS_URL, ping_interval=None) as ws:
                _ws_bitget = ws
                logger.info("Bitget WS connected")
                for sub in subs:
                    await ws.send(json.dumps(sub))
                async for message in ws:
                    if _stop_event.is_set():
                        break
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
                        sym_fmt = format_bitget_symbol(sym)
                        with _lock:
                            book = _liquidity[sym_fmt]
                            book["bids"] = {
                                float(b[0]): float(b[1]) for b in item.get("bids", [])
                            }
                            book["asks"] = {
                                float(a[0]): float(a[1]) for a in item.get("asks", [])
                            }
        except (websockets.ConnectionClosed, asyncio.TimeoutError) as exc:
            if _stop_event.is_set():
                break
            logger.warning(
                "Bitget WS disconnected: %s. Reconnecting in %ss...",
                exc,
                RECONNECT_DELAY,
            )
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as exc:
            logger.error("Unexpected error in Bitget listener: %s", exc, exc_info=True)
            if not _stop_event.is_set():
                await asyncio.sleep(RECONNECT_DELAY)
    _ws_bitget = None

async def _run(symbols):
    tasks = [
        asyncio.create_task(_binance_listener(symbols)),
        asyncio.create_task(_bitget_listener(symbols)),
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("Critical error in liquidity_ws _run: %s", exc, exc_info=True)
        raise


def _start_loop(loop: asyncio.AbstractEventLoop, symbols: Iterable[str]):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run(symbols))
    except Exception as exc:
        logger.error("Error in liquidity_ws event loop: %s", exc, exc_info=True)
    finally:
        loop.close()
        global _loop
        _loop = None
        logger.info("Liquidity WS event loop closed")


def start(symbols: Iterable[str] | None = None):
    """Start or update websocket listeners in a background thread."""
    global TOP_15_SYMBOLS
    if symbols is None:
        TOP_15_SYMBOLS = get_top_15_symbols()
        symbols = TOP_15_SYMBOLS
    global _loop, _thread
    _stop_event.clear()
    if _thread and not _thread.is_alive():
        logger.info("Liquidity WS thread dead, resetting loop")
        _loop = None
        _thread = None
    if _loop is not None:
        logger.info("Liquidity WS already running")
        return
    _loop = asyncio.new_event_loop()
    _thread = threading.Thread(target=_start_loop, args=(_loop, symbols), daemon=True)
    _thread.start()
    logger.info("Liquidity WS thread started")


def stop():
    """Stop the websocket listeners and event loop."""
    global _loop, _thread, _ws_binance, _ws_bitget
    if _loop is None:
        return
    logger.info("Stopping Liquidity WS...")
    _stop_event.set()
    if _ws_binance is not None:
        asyncio.run_coroutine_threadsafe(_ws_binance.close(), _loop)
    if _ws_bitget is not None:
        asyncio.run_coroutine_threadsafe(_ws_bitget.close(), _loop)
    _loop.call_soon_threadsafe(_loop.stop)
    if _thread:
        _thread.join(timeout=5)
    _loop = None
    _thread = None
    _ws_binance = None
    _ws_bitget = None


def get_liquidity(symbol=None):
    """Get current liquidity map. If symbol provided, return its book."""
    with _lock:
        if symbol:
            if symbol in _liquidity:
                return _liquidity.get(symbol)
            return None
        return {k: v.copy() for k, v in _liquidity.items()}


# Example usage:
# start()  # begins streaming liquidity for TOP_15_SYMBOLS

