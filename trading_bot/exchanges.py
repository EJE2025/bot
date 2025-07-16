import ccxt
import logging
import time
from . import config

logger = logging.getLogger(__name__)

_EXCHANGE_CACHE = {}


class MockExchange:
    """Simple deterministic mock of a futures exchange for tests."""

    def __init__(self):
        # Minimal market information so the bot validations succeed
        self.markets = {
            "BTC/USDT:USDT": {"id": "BTCUSDT_UMCBL", "contractSize": 1},
            "ETH/USDT:USDT": {"id": "ETHUSDT_UMCBL", "contractSize": 1},
        }

        # Internal state
        self.positions: list[dict] = []
        self.open_orders: list[dict] = []
        self.leverage: dict[str, int] = {}
        self.balance = {"USDT": {"free": 100_000, "used": 0, "total": 100_000}}
        self.last_order_id = 0
        # control whether limit orders fill immediately
        self.order_status_flow = "filled"  # or "open"

    # --- Utility helpers -------------------------------------------------

    def _next_order_id(self) -> str:
        self.last_order_id += 1
        return f"MOCK_{self.last_order_id}"

    def _get_market_price(self, symbol: str) -> float:
        # extremely naive deterministic price generator
        if "BTC" in symbol:
            return 30_000.0
        if "ETH" in symbol:
            return 2_000.0
        return 1.0

    def _simulate_fill(self, order: dict) -> None:
        """Mark order as filled and update positions/balance."""
        order["status"] = "closed"
        fill_price = order.get("price") or self._get_market_price(order["symbol"])
        order["average"] = fill_price
        amount = order["amount"]
        side = order["side"]
        sym = order["symbol"]

        # adjust balance (simplified: 1 contract = 1 quote currency)
        bal = self.balance["USDT"]
        cost = amount * fill_price
        if side.lower() == "buy" or side.lower() == "close_short":
            bal["used"] += cost
            bal["free"] -= cost
        else:
            bal["free"] += cost

        # manage positions
        pos = next((p for p in self.positions if p["symbol"] == sym), None)
        if side.lower() in ("buy", "close_short"):
            direction = "long"
        else:
            direction = "short"
        if pos:
            pos["contracts"] += amount
            pos["entryPrice"] = fill_price
            pos["side"] = direction
        else:
            self.positions.append({
                "symbol": sym,
                "contracts": amount,
                "entryPrice": fill_price,
                "side": direction,
                "leverage": self.leverage.get(sym, 1),
            })

    # --- ccxt-like API ---------------------------------------------------

    def load_markets(self):
        return self.markets

    def fetch_balance(self):
        return self.balance

    def set_leverage(self, leverage: int, market_id: str):
        self.leverage[market_id] = leverage
        return {"info": f"Leverage set to {leverage}x for {market_id}"}

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        order = {
            "id": self._next_order_id(),
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "open",
            "timestamp": int(time.time() * 1000),
        }
        self.open_orders.append(order)

        # orders fill instantly for simplicity unless configured otherwise
        if self.order_status_flow == "filled" or type == "market":
            self._simulate_fill(order)
        return order

    def cancel_order(self, order_id, symbol=None):
        for order in self.open_orders:
            if order["id"] == order_id:
                order["status"] = "canceled"
        self.open_orders = [o for o in self.open_orders if o["status"] == "open"]
        return {"id": order_id, "status": "canceled"}

    def fetch_order(self, order_id, symbol=None):
        for order in self.open_orders:
            if order["id"] == order_id:
                return order
        # assume closed if not found
        return {"id": order_id, "status": "closed", "average": self._get_market_price(symbol) if symbol else None}

    def fetch_open_orders(self):
        return [o for o in self.open_orders if o.get("status") == "open"]

    def fetch_positions(self, params=None):
        return list(self.positions)

    def market(self, symbol):
        return {"id": symbol}

def get_exchange(name: str):
    name = name.lower()
    if name in _EXCHANGE_CACHE:
        return _EXCHANGE_CACHE[name]

    if config.TEST_MODE:

        logger.warning("TEST_MODE enabled - using MockExchange")
        ex = MockExchange()
    elif name == "bitget":
        ex = ccxt.bitget({
            "apiKey": config.BITGET_API_KEY,
            "secret": config.BITGET_API_SECRET,
            "password": config.BITGET_PASSPHRASE,
            "options": {"defaultType": "swap"},
        })
    elif name == "binance":
        ex = ccxt.binance({
            "apiKey": config.BINANCE_API_KEY,
            "secret": config.BINANCE_API_SECRET,
            "options": {"defaultType": "future"},
        })
    elif name == "mexc":
        ex = ccxt.mexc({
            "apiKey": config.MEXC_API_KEY,
            "secret": config.MEXC_API_SECRET,
        })
    else:
        raise ValueError(f"Unsupported exchange {name}")
    try:
        ex.load_markets()
    except Exception as exc:
        logger.error("Failed to connect to %s: %s", name, exc)
        ex = MockExchange()
        ex.load_markets()
        logger.error("Using MockExchange due to connection failure")
    _EXCHANGE_CACHE[name] = ex
    return ex
