import logging
import time
from . import config
from .exchanges import get_exchange

logger = logging.getLogger(__name__)

exchange = get_exchange(config.DEFAULT_EXCHANGE)


class OrderSubmitError(Exception):
    pass


def setup_leverage(symbol_raw: str, leverage: int) -> None:
    bitget_sym = symbol_raw.replace("_", "") + "_UMCBL"
    if bitget_sym not in exchange.markets:
        logger.warning("Skip leverage: %s not available", bitget_sym)
        return
    try:
        market_id = exchange.market(bitget_sym)["id"]
        exchange.set_leverage(leverage, market_id)
        logger.info("Leverage set to %dx for %s", leverage, bitget_sym)
    except Exception as exc:
        logger.error("Leverage error on %s: %s", bitget_sym, exc)


def open_position(symbol: str, side: str, amount: float, price: float,
                  order_type: str = "limit", stop_price: float | None = None):
    bitget_sym = symbol.replace("_", "/") + ":USDT"
    if bitget_sym not in exchange.markets:
        raise OrderSubmitError(f"Market {bitget_sym} not available")
    for attempt in range(3):
        try:
            params = {"timeInForce": "GTC", "holdSide": "long" if side == "BUY" else "short"}
            ord_type = order_type.lower()
            ord_price = price
            if ord_type == "market":
                ord_type = "market"
                ord_price = None
            elif ord_type == "stop":
                params["stopPrice"] = stop_price or price
                ord_type = "limit"
            else:
                ord_type = "limit"
            return exchange.create_order(
                symbol=bitget_sym,
                type=ord_type,
                side="buy" if side == "BUY" else "sell",
                amount=amount,
                price=ord_price,
                params=params,
            )
        except Exception:
            time.sleep(2 ** attempt)
    raise OrderSubmitError(f"Failed to open position {symbol}")


def close_position(symbol: str, side: str, amount: float, order_type: str = "market"):
    bitget_sym = symbol.replace("_", "/") + ":USDT"
    if bitget_sym not in exchange.markets:
        raise OrderSubmitError(f"Market {bitget_sym} not available")
    for attempt in range(3):
        try:
            params = {"reduceOnly": True}
            ord_type = order_type.lower()
            ord_price = None
            if ord_type not in ("market", "limit"):
                ord_type = "market"
            return exchange.create_order(
                symbol=bitget_sym,
                type=ord_type,
                side="buy" if side == "close_short" else "sell",
                amount=amount,
                price=ord_price,
                params=params,
            )
        except Exception:
            time.sleep(2 ** attempt)
    raise OrderSubmitError(f"Failed to close position {symbol}")


def cancel_order(order_id: str, symbol: str):
    bitget_sym = symbol.replace("_", "/") + ":USDT"
    try:
        exchange.cancel_order(order_id, bitget_sym)
    except Exception as exc:
        logger.error("Error canceling %s: %s", order_id, exc)
