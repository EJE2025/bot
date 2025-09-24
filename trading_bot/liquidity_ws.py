import asyncio
import json
import logging
import threading
from collections import defaultdict
from typing import Iterable, Dict, Any


import websockets
from . import config, data, exchanges

logger = logging.getLogger(__name__)

DEPTH = 10
BINANCE_INTERVAL_MS = 500
RECONNECT_DELAY = 5

DEFAULT_TOP_15_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
    "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LTCUSDT",
    "BCHUSDT", "LINKUSDT", "UNIUSDT", "ALGOUSDT", "ATOMUSDT",
]


class LiquidityStream:
    """Asynchronous producer/consumer to track a single exchange order book."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._orderbook: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False

    async def _producer(self, symbols: list[str], url: str):
        params = [f"{s.lower()}@depth" for s in symbols]
        async with websockets.connect(f"{url}?streams={'/'.join(params)}") as ws:
            self._ws = ws
            self._running = True
            async for message in ws:
                await self._queue.put(message)
                if not self._running:
                    break

    async def _consumer(self):
        while self._running:
            message = await self._queue.get()
            if message is None:
                break
            async with self._lock:
                self._process(json.loads(message))

    def _process(self, data: dict) -> None:
        symbol = data.get("symbol") or data.get("stream", "").split("@")[0].upper()
        bids = data.get("bids") or data.get("data", {}).get("bids", [])
        asks = data.get("asks") or data.get("data", {}).get("asks", [])
        self._orderbook[symbol] = {
            "bids": [[float(p), float(q)] for p, q in bids],
            "asks": [[float(p), float(q)] for p, q in asks],
        }

    async def listen(self, symbols: list[str], url: str = "wss://stream.binance.com:9443/stream"):
        self._running = True
        producer = asyncio.create_task(self._producer(symbols, url))
        consumer = asyncio.create_task(self._consumer())
        await asyncio.gather(producer, consumer)

    async def stop(self) -> None:
        self._running = False
        await self._queue.put(None)
        if self._ws is not None:
            await self._ws.close()

    def get_orderbook(self, symbol: str) -> dict | None:
        return self._orderbook.get(symbol.upper())


def get_top_15_symbols() -> list[str]:

    ex = exchanges.get_exchange(config.DEFAULT_EXCHANGE)
    symbols = data.get_common_top_symbols(ex, 15)
    if not symbols:
        return DEFAULT_TOP_15_SYMBOLS
    return [s.replace("_", "") for s in symbols]


def format_binance_symbol(sym_raw: str) -> str:
    base = sym_raw[:-4]
    quote = sym_raw[-4:]
    return f"{base}_{quote} (Binance)"


def format_binance_stream_symbol(sym_raw: str) -> str:
    return sym_raw.lower() + f"@depth{DEPTH}@{BINANCE_INTERVAL_MS}ms"


def format_bitget_symbol(sym: str) -> str:
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
    cleaned = sym.replace("/", "").replace("_", "").upper()
    if cleaned.endswith("USDT") and len(cleaned) > 4:
        return f"{cleaned[:-4]}-USDT"
    return cleaned


class DualExchangeLiquidityStream:
    """Maintain Binance and Bitget order books with automatic reconnection."""

    def __init__(self) -> None:
        self._orderbook: Dict[str, Dict[str, Dict[float, float]]] = defaultdict(lambda: {"bids": {}, "asks": {}})
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws_binance: websockets.WebSocketClientProtocol | None = None
        self._ws_bitget: websockets.WebSocketClientProtocol | None = None

    def _ws_targets(self) -> list[str]:
        mode = (config.WS_EXCHANGE or config.PRIMARY_EXCHANGE).lower()
        targets: list[str] = []
        if mode in {"bitget", "both"} or (mode not in {"binance", "bitget"} and not config.ENABLE_BINANCE):
            if config.ENABLE_BITGET:
                targets.append("bitget")
        if mode in {"binance", "both"} and config.ENABLE_BINANCE:
            targets.append("binance")
        return targets

    async def _binance_listener(self, symbols: Iterable[str]):
        streams = "/".join(format_binance_stream_symbol(s) for s in symbols)
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to Binance WS: %s", url)
                async with websockets.connect(url, ping_interval=None) as ws:
                    self._ws_binance = ws
                    logger.info("Binance WS connected")
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                        data = json.loads(message)
                        content = data.get("data")
                        if not content or "bids" not in content:
                            continue
                        sym_raw = content.get("s")
                        if not sym_raw:
                            continue
                        sym = format_binance_symbol(sym_raw)
                        async with self._lock:
                            book = self._orderbook[sym]
                            book["bids"] = {float(p): float(q) for p, q in content.get("bids", [])}
                            book["asks"] = {float(p): float(q) for p, q in content.get("asks", [])}
            except (websockets.ConnectionClosed, asyncio.TimeoutError) as exc:
                if self._stop_event.is_set():
                    break
                logger.warning("Binance WS disconnected: %s. Reconnecting in %ss...", exc, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)
            except Exception as exc:
                logger.error("Unexpected error in Binance listener: %s", exc, exc_info=True)
                if not self._stop_event.is_set():
                    await asyncio.sleep(RECONNECT_DELAY)
        self._ws_binance = None

    async def _bitget_listener(self, symbols: Iterable[str]):
        subs = [{"op": "subscribe", "args": [f"books{DEPTH}:{format_bitget_stream_symbol(s)}"]} for s in symbols]
        url = "wss://ws.bitget.com/mix/v1/stream"
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to Bitget WS: %s", url)
                async with websockets.connect(url, ping_interval=None) as ws:
                    self._ws_bitget = ws
                    logger.info("Bitget WS connected")
                    for sub in subs:
                        await ws.send(json.dumps(sub))
                    async for message in ws:
                        if self._stop_event.is_set():
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
                            async with self._lock:
                                book = self._orderbook[sym_fmt]
                                book["bids"] = {float(b[0]): float(b[1]) for b in item.get("bids", [])}
                                book["asks"] = {float(a[0]): float(a[1]) for a in item.get("asks", [])}
            except (websockets.ConnectionClosed, asyncio.TimeoutError) as exc:
                if self._stop_event.is_set():
                    break
                logger.warning("Bitget WS disconnected: %s. Reconnecting in %ss...", exc, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)
            except Exception as exc:
                logger.error("Unexpected error in Bitget listener: %s", exc, exc_info=True)
                if not self._stop_event.is_set():
                    await asyncio.sleep(RECONNECT_DELAY)
        self._ws_bitget = None

    async def _run(self, symbols: Iterable[str]):
        tasks: list[asyncio.Task] = []
        targets = self._ws_targets()
        if "binance" in targets:
            tasks.append(asyncio.create_task(self._binance_listener(symbols)))
        if "bitget" in targets:
            tasks.append(asyncio.create_task(self._bitget_listener(symbols)))
        if not tasks:
            logger.warning("No websocket targets enabled; liquidity stream idle")
            return
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    def _start_loop(self, symbols: Iterable[str]):
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run(symbols))
        finally:
            self._loop.close()
            self._loop = None
            logger.info("Liquidity WS event loop closed")

    def start(self, symbols: Iterable[str] | None = None):
        if self._loop is not None:
            logger.info("Liquidity WS already running")
            return
        if symbols is None:
            if config.BOT_MODE == "shadow" and config.SYMBOLS:
                symbols = config.SYMBOLS
            else:
                symbols = get_top_15_symbols()
        self._stop_event.clear()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._start_loop,
            args=(list(symbols),),
            daemon=True,
        )
        self._thread.start()
        logger.info("Liquidity WS thread started")

    def stop(self):
        if self._loop is None:
            return
        logger.info("Stopping Liquidity WS...")
        self._stop_event.set()
        if self._ws_binance is not None:
            asyncio.run_coroutine_threadsafe(
                self._ws_binance.close(),
                self._loop,
            )
        if self._ws_bitget is not None:
            asyncio.run_coroutine_threadsafe(
                self._ws_bitget.close(),
                self._loop,
            )
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        self._ws_binance = None
        self._ws_bitget = None

    async def get_orderbook(self, symbol: str | None = None) -> Dict[str, Any]:
        async with self._lock:
            if symbol:
                return self._orderbook.get(symbol, {})
            return {k: v.copy() for k, v in self._orderbook.items()}


_stream: DualExchangeLiquidityStream | None = None


def start(symbols: Iterable[str] | None = None):
    global _stream
    if _stream is None:
        _stream = DualExchangeLiquidityStream()
    _stream.start(symbols)


def stop():
    if _stream:
        _stream.stop()


def get_liquidity(symbol: str | None = None) -> Dict[str, Any]:
    if _stream is None or _stream._loop is None:
        return {}
    fut = asyncio.run_coroutine_threadsafe(
        _stream.get_orderbook(symbol),
        _stream._loop,
    )
    return fut.result()
