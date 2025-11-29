import ccxt
import logging
import time
import requests

from .. import config

logger = logging.getLogger(__name__)

_EXCHANGE_CACHE = {}


class MockExchange:
    """Simple deterministic mock of a futures exchange for tests.

    Parameters
    ----------
    slippage : float, optional
        Percentual slippage applied to market prices (e.g. ``0.01`` for 1%).
    fee_rate : float, optional
        Trading fee rate applied on the notional value.
    order_status_flow : str, optional
        Control order life cycle. ``"filled"`` fills orders immediately while
        ``"open"`` leaves limit orders pending.
    Notes
    -----
    The mock generates a fixed set of markets with deterministic volumes so the
    most liquid pairs are stable across runs.
    """

    def __init__(self, slippage: float = 0.0, fee_rate: float = 0.0,
                 order_status_flow: str = "filled"):
        self.slippage = slippage
        self.fee_rate = fee_rate
        self.order_status_flow = order_status_flow

        # Generate a deterministic set of markets with descending volume
        bases = [
            "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "AVAX",
            "MATIC", "LTC", "BCH", "LINK", "UNI", "ALGO", "ATOM", "FIL",
            "APT", "ARB", "OP", "SUI", "PEPE", "WIF", "FLOKI", "BONK", "MEME",
        ]
        self.markets = {}
        for idx, base in enumerate(bases):
            sym = f"{base}/USDT:USDT"
            self.markets[sym] = {
                "id": sym.replace("/", "").replace(":USDT", "") + "_UMCBL",
                "contractSize": 1,
                "symbol": sym,
                # decreasing volume so the first symbols are always top ranked
                "volume": 1_000_000 - idx * 1000,
            }

        # Internal state
        self.positions: list[dict] = []
        self.open_orders: list[dict] = []
        self.leverage: dict[str, int] = {}
        self.balance = {"USDT": {"free": 100_000, "used": 0, "total": 100_000}}
        self.last_order_id = 0

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
        side = order["side"].lower()

        # apply simple slippage model
        if self.slippage:
            if side in ("buy", "close_short"):
                fill_price *= 1 + self.slippage
            else:
                fill_price *= 1 - self.slippage

        order["average"] = fill_price
        amount = order["amount"]
        order["filled"] = float(amount)
        order["remaining"] = 0.0
        sym = order["symbol"]

        # adjust balance (simplified: 1 contract = 1 quote currency)
        bal = self.balance["USDT"]
        cost = amount * fill_price
        fee = cost * self.fee_rate
        total = cost + fee

        if side == "buy":
            bal["used"] += total
            bal["free"] -= total
        elif side == "sell":
            bal["free"] -= total
        elif side == "close_long":
            bal["used"] -= total
            bal["free"] += total
        elif side == "close_short":
            bal["free"] += total

        # manage positions
        pos = next((p for p in self.positions if p["symbol"] == sym), None)

        if side == "close_long" and pos and pos.get("side") == "long":
            pos["contracts"] -= amount
            if pos["contracts"] <= 0:
                self.positions.remove(pos)
            return
        if side == "close_short" and pos and pos.get("side") == "short":
            pos["contracts"] -= amount
            if pos["contracts"] <= 0:
                self.positions.remove(pos)
            return

        if side in ("buy", "close_short"):
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

    def fetch_markets(self):
        """Return market definitions including simulated volume."""
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
            "filled": 0.0,
            "remaining": float(amount),
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
                order["remaining"] = max(
                    0.0,
                    float(order.get("amount", 0.0)) - float(order.get("filled", 0.0)),
                )
        self.open_orders = [o for o in self.open_orders if o["status"] == "open"]
        return {"id": order_id, "status": "canceled"}

    def fetch_order(self, order_id, symbol=None):
        for order in self.open_orders:
            if order["id"] == order_id:
                return order
        # assume closed if not found
        price = self._get_market_price(symbol) if symbol else None
        return {
            "id": order_id,
            "status": "closed",
            "average": price,
            "amount": None,
            "filled": None,
            "remaining": 0.0,
        }

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

    if not isinstance(ex, MockExchange) and name == "bitget":
        def fetch_markets():
            url = "https://api.bitget.com/api/v2/mix/market/tickers"
            params = {"productType": "USDT-FUTURES"}
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != "00000" or "data" not in data:
                    raise RuntimeError("Bad response")
                markets = {}
                for item in data["data"]:
                    sym = item.get("symbol", "")
                    if not sym.endswith("USDT") or len(sym) <= 4:
                        continue
                    base = sym[:-4]
                    quote = sym[-4:]
                    unified = f"{base}/{quote}:USDT"
                    markets[unified] = {
                        "symbol": unified,
                        "volume": float(item.get("usdtVolume") or item.get("quoteVolume", 0)),
                    }
                return markets
            except Exception as exc:
                logger.error("Error fetching Bitget markets: %s", exc)
                return {sym: {"symbol": sym, "volume": 0.0} for sym in ex.markets.keys()}

        ex.fetch_markets = fetch_markets
    _EXCHANGE_CACHE[name] = ex
    return ex
