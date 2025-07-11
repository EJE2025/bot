import logging
import time
from . import config
from .exchanges import get_exchange

logger = logging.getLogger(__name__)

try:
    exchange = get_exchange(config.DEFAULT_EXCHANGE)
except Exception as exc:
    logger.error("Exchange initialization failed: %s", exc)
    exchange = None


class OrderSubmitError(Exception):
    pass



def fetch_positions():
    """Return a list of current Bitget positions."""
    if exchange is None:
        return []
    try:
        positions = exchange.fetch_positions(params={"productType": "USDT-FUTURES"})
        return [p for p in positions if float(p.get("contracts", 0)) != 0]
    except Exception as exc:
        logger.error("Failed fetching positions: %s", exc)
        return []


def fetch_balance():
    """Return available USDT balance if possible."""
    if exchange is None:
        return 0.0
    try:
        bal = exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        return usdt.get("free", 0.0)
    except Exception as exc:
        logger.error("Balance fetch error: %s", exc)
        return 0.0


def check_order_filled(order_id: str, symbol: str, timeout: int = 15) -> bool:
    """Poll order status until filled or timeout."""
    if exchange is None:
        return False
    bitget_sym = symbol.replace("_", "/") + ":USDT"
    end = time.time() + timeout
    while time.time() < end:
        try:
            info = exchange.fetch_order(order_id, bitget_sym)
            status = info.get("status")
            if status in ("closed", "filled"):
                return True
            if status in ("canceled", "rejected"):
                return False
        except Exception:
            pass
        time.sleep(2)
    return False



def setup_leverage(symbol_raw: str, leverage: int) -> None:
    if exchange is None:
        logger.error("Exchange not initialized")
        return
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
    if exchange is None:
        raise OrderSubmitError("Exchange not initialized")
    bitget_sym = symbol.replace("_", "/") + ":USDT"
    if bitget_sym not in exchange.markets:
        raise OrderSubmitError(f"Market {bitget_sym} not available")

    bal = fetch_balance()
    cost = amount * price / exchange.markets[bitget_sym].get("contractSize", 1)
    if bal < cost:
        raise OrderSubmitError("Insufficient balance")

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

            order = exchange.create_order(

                symbol=bitget_sym,
                type=ord_type,
                side="buy" if side == "BUY" else "sell",
                amount=amount,
                price=ord_price,
                params=params,
            )

            return order

        except Exception:
            time.sleep(2 ** attempt)
    raise OrderSubmitError(f"Failed to open position {symbol}")


def close_position(symbol: str, side: str, amount: float, order_type: str = "market"):
    if exchange is None:
        raise OrderSubmitError("Exchange not initialized")
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
    if exchange is None:
        logger.error("Exchange not initialized")
        return
    try:
        exchange.cancel_order(order_id, bitget_sym)
    except Exception as exc:
        logger.error("Error canceling %s: %s", order_id, exc)
