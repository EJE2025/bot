import ccxt
import logging
from . import config

logger = logging.getLogger(__name__)

_EXCHANGE_CACHE = {}


class MockExchange:
    """Fallback exchange used when real connectivity fails."""

    markets: dict = {}

    def load_markets(self):
        self.markets = {}

    def create_order(self, *args, **kwargs):
        logger.info("Mock order created %s %s", args, kwargs)
        return {"id": "MOCK"}

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
    _EXCHANGE_CACHE[name] = ex
    return ex
