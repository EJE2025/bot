import ccxt
from . import config

_EXCHANGE_CACHE = {}

def get_exchange(name: str):
    name = name.lower()
    if name in _EXCHANGE_CACHE:
        return _EXCHANGE_CACHE[name]

    if name == "bitget":
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
    ex.load_markets()
    _EXCHANGE_CACHE[name] = ex
    return ex
