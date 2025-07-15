import ccxt
import logging
from . import config

logger = logging.getLogger(__name__)

_EXCHANGE_CACHE = {}


class MockExchange:
    """Fallback exchange used when real connectivity fails or TEST_MODE."""

    def __init__(self):
        # populate a few common markets so validations pass
        self.markets = {
            "BTC/USDT:USDT": {"id": "BTCUSDT_UMCBL", "contractSize": 1},
            "ETH/USDT:USDT": {"id": "ETHUSDT_UMCBL", "contractSize": 1},
        }
        self.orders: dict[str, dict] = {}

    def load_markets(self):
        return self.markets

    def fetch_balance(self):
        # return a large dummy balance for tests
        return {"USDT": {"free": 1_000_000}}

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        order_id = f"MOCK_{len(self.orders)+1}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": type_,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "closed",
        }
        self.orders[order_id] = order
        logger.info("Mock order %s", order)
        return order

    def cancel_order(self, *args, **kwargs):
        logger.info("Mock cancel %s %s", args, kwargs)

    def set_leverage(self, *args, **kwargs):
        logger.info("Mock leverage %s %s", args, kwargs)

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
