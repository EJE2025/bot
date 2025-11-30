import json
import time
import hmac
import base64
import asyncio
import json
import websockets


BITGET_WS_URL = "wss://ws.bitget.com/mix/v1/stream"


class BitgetWebSocket:
    def __init__(self, api_key, api_secret, passphrase, trade_manager, logger):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.passphrase = passphrase
        self.trade_manager = trade_manager
        self.logger = logger
        self._last_reconnect_sync: float = 0.0

    def _signature(self, timestamp):
        msg = f"{timestamp}GET/user/stream".encode()
        sign = hmac.new(self.api_secret, msg, digestmod="sha256").digest()
        return base64.b64encode(sign).decode()

    async def _auth(self, ws):
        ts = str(int(time.time()))
        sign = self._signature(ts)

        payload = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": ts,
                "sign": sign
            }]
        }

        await ws.send(json.dumps(payload))
        self.logger.info("Bitget WS: Authenticating...")

    async def subscribe(self, ws):
        channels = [
            {"instType": "mc", "channel": "orders"},
            {"instType": "mc", "channel": "positions"}
        ]

        for c in channels:
            await ws.send(json.dumps({"op": "subscribe", "args": [c]}))

        self.logger.info("Bitget WS: Subscribed to orders + positions")

    async def _resync_after_reconnect(self) -> None:
        """Perform a REST snapshot after reconnecting to avoid drift."""

        now = time.time()
        if now - self._last_reconnect_sync < 5:
            return

        try:
            await asyncio.to_thread(self.trade_manager.reconcile_positions)
            self._last_reconnect_sync = now
            self.logger.info("Bitget WS: State reconciled after reconnect")
        except Exception as exc:
            self.logger.error("Bitget WS: Failed to reconcile after reconnect: %s", exc)

    async def handler(self, data):
        if "data" not in data:
            return

        for entry in data["data"]:
            channel = entry.get("channel")

            if channel == "orders":
                await self._handle_order(entry)

            elif channel == "positions":
                await self._handle_position(entry)

    async def _handle_order(self, order):
        symbol = order["symbol"]
        status = order["status"]
        side = str(order.get("side", "")).lower()

        if "close" in side and status in {"filled", "partial"}:
            self.logger.debug("WS: close order event ignored for %s (awaiting position feed)", symbol)
            return

        if status == "filled":
            self.trade_manager.ws_order_filled(order)
            self.logger.info(f"WS: order filled {symbol}")

        elif status == "partial":
            self.trade_manager.ws_order_partial(order)
            self.logger.info(f"WS: order partial {symbol}")

        elif status == "cancelled":
            self.trade_manager.ws_order_cancelled(order)
            self.logger.warning(f"WS: order cancelled {symbol}")

    async def _handle_position(self, pos):
        symbol = pos["symbol"]
        size = float(pos.get("holdVolume", 0))

        if size != 0:
            self.trade_manager.ws_position_update(pos)
            self.logger.info(f"WS: position update {symbol}, size={size}")
        else:
            self.trade_manager.ws_position_closed(pos)
            self.logger.info(f"WS: position closed {symbol}")

    async def run(self):
        while True:
            try:
                async with websockets.connect(BITGET_WS_URL, ping_interval=20) as ws:

                    await self._auth(ws)
                    await self.subscribe(ws)
                    await self._resync_after_reconnect()

                    while True:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        await self.handler(msg)

            except Exception as e:
                self.logger.error(f"Bitget WS error: {e}")
                await asyncio.sleep(3)
