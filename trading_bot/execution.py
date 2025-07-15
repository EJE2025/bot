import logging
import time
from . import config
from .exchanges import get_exchange, MockExchange



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


def fetch_open_orders():
    """Return a list of open orders on the exchange."""
    if exchange is None:
        return []
    try:
        orders = exchange.fetch_open_orders()
        return orders
    except Exception as exc:
        logger.error("Failed fetching open orders: %s", exc)
        return []


def fetch_balance():
    """Return available USDT balance if possible."""
    if exchange is None:
        return 0.0
    if config.TEST_MODE or isinstance(exchange, MockExchange):
        return 1_000_000.0

    try:
        bal = exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        return usdt.get("free", 0.0)
    except Exception as exc:
        logger.error("Balance fetch error: %s", exc)
        return 0.0



def check_order_filled(order_id: str, symbol: str, timeout: int = config.ORDER_FILL_TIMEOUT) -> bool:

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
                logger.warning("Order %s %s", order_id, status)
                return False
        except Exception as exc:
            logger.error("Fetch order %s error: %s", order_id, exc)
        time.sleep(2)
    logger.warning("Order %s not filled after %ds", order_id, timeout)
    return False



def setup_leverage(exchange, symbol_raw: str, leverage: int = 10) -> bool:
    """Configure leverage for a symbol and confirm on Bitget."""
    if exchange is None:
        logger.error("Exchange not initialized")
        return False
    if config.TEST_MODE or isinstance(exchange, MockExchange):
        logger.info("Mock leverage setup for %s", symbol_raw)
        return True

    # Build unified symbol used by ccxt: "BTC/USDT:USDT"
    base = symbol_raw[:-4]
    quote = symbol_raw[-4:]
    unified = f"{base}/{quote}:USDT"

    if unified not in exchange.markets:
        logger.warning(
            "Skip leverage: %s no encontrado en mercados como %s", symbol_raw, unified
        )
        return False

    market_id = exchange.markets[unified]["id"]
    try:
        exchange.set_leverage(leverage, market_id)
        logger.info("Leverage set to %dx for %s", leverage, market_id)
        # verify leverage was applied
        positions = exchange.fetch_positions([market_id])
        for pos in positions:
            if pos.get("info", {}).get("symbol") == market_id or pos.get("symbol") == market_id:
                applied = float(pos.get("leverage", 0))
                logger.info("Confirmed leverage for %s: %sx", market_id, applied)
                return applied == leverage
        logger.warning("Could not confirm leverage for %s", market_id)
        return False
    except Exception as e:
        logger.error("Error setting leverage for %s: %s", market_id, e)
        return False


def open_position(symbol: str, side: str, amount: float, price: float,
                  order_type: str = "limit", stop_price: float | None = None):
    if exchange is None:
        raise OrderSubmitError("Exchange not initialized")
    bitget_sym = symbol.replace("_", "/") + ":USDT"

    if not (config.TEST_MODE or isinstance(exchange, MockExchange)):
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
        except Exception as exc:
            if attempt < 2:
                logger.warning("Retry open %s attempt %d error: %s", symbol, attempt + 1, exc)
                time.sleep(2 ** attempt)
            else:
                raise OrderSubmitError(f"Failed to open position {symbol}") from exc

    raise OrderSubmitError(f"Failed to open position {symbol}")


def close_position(symbol: str, side: str, amount: float, order_type: str = "market"):
    if exchange is None:
        raise OrderSubmitError("Exchange not initialized")
    bitget_sym = symbol.replace("_", "/") + ":USDT"

    if not (config.TEST_MODE or isinstance(exchange, MockExchange)):
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

        except Exception as exc:
            if attempt < 2:
                logger.warning("Retry close %s attempt %d error: %s", symbol, attempt + 1, exc)
                time.sleep(2 ** attempt)
            else:
                raise OrderSubmitError(f"Failed to close position {symbol}") from exc

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



def cleanup_old_orders(max_age: int = config.ORDER_MAX_AGE):
    """Cancel pending orders older than max_age seconds."""
    now = time.time()
    for order in fetch_open_orders():
        ts = order.get("timestamp")
        if not ts:
            continue
        age = now - ts / 1000
        if age > max_age:
            oid = order.get("id")
            sym = order.get("symbol", "")
            sym_clean = sym.replace("/", "_").replace(":USDT", "")
            logger.info("Cancelling stale order %s for %s age %.1fs", oid, sym_clean, age)
            cancel_order(oid, sym_clean)
