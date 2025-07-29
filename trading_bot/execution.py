import logging
import time
import asyncio
import ccxt
from . import config
from .exchanges import get_exchange, MockExchange
from .utils import circuit_breaker



logger = logging.getLogger(__name__)

# Limit concurrent API requests across async tasks
api_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

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



def setup_leverage(exchange, symbol: str, leverage: int = 10):
    """Configure leverage for a symbol and confirm on Bitget."""
    if exchange is None:
        logger.error("Exchange not initialized")
        return {"success": False, "error": "exchange not initialized"}

    # Build unified ccxt symbol like "BTC/USDT:USDT" from raw symbol if needed
    if "/" not in symbol:
        base = symbol[:-4]
        quote = symbol[-4:]
        symbol = f"{base}/{quote}:USDT"

    if config.TEST_MODE or isinstance(exchange, MockExchange):
        logger.info("Mock leverage setup for %s", symbol)
        try:
            result = exchange.set_leverage(leverage, symbol)
            return {"success": True, "symbol": symbol, "result": result}
        except Exception as exc:  # pragma: no cover - mock rarely fails
            logger.error("Error setting leverage for %s: %s", symbol, exc)
            return {"success": False, "symbol": symbol, "error": str(exc)}

    if symbol not in exchange.markets:
        logger.warning(
            "Skip leverage: %s no encontrado en mercados", symbol
        )
        return {"success": False, "symbol": symbol, "error": "symbol not found"}

    try:
        result = exchange.set_leverage(leverage, symbol)
        logger.info("Leverage set to %dx for %s", leverage, symbol)
        # verify leverage was applied
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos.get("symbol") == symbol or pos.get("info", {}).get("symbol") == symbol:
                applied = float(pos.get("leverage", 0))
                logger.info("Confirmed leverage for %s: %sx", symbol, applied)
                return {"success": applied == leverage, "symbol": symbol, "result": result}
        logger.warning("Could not confirm leverage for %s", symbol)
        return {"success": False, "symbol": symbol, "result": result}
    except ccxt.BaseError as exc:
        logger.error("API error setting leverage for %s: %s", symbol, exc)
        return {"success": False, "symbol": symbol, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("Error setting leverage for %s: %s", symbol, exc)
        return {"success": False, "symbol": symbol, "error": str(exc)}


@circuit_breaker(fallback=None, max_failures=3, reset_timeout=60)
def open_position(symbol: str, side: str, amount: float, price: float,
                  order_type: str = "limit", stop_price: float | None = None):
    """Send an order to open a position and return the order data.

    The function validates that the exchange returns a dictionary containing the
    keys ``id``, ``status`` and ``average``. Temporary errors are retried up to
    ``ORDER_SUBMIT_ATTEMPTS`` times with exponential backoff.
    """
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
    max_attempts = config.ORDER_SUBMIT_ATTEMPTS
    for attempt in range(max_attempts):
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
            if not isinstance(order, dict) or not {"id", "status", "average"}.issubset(order.keys()):
                raise OrderSubmitError(f"Unexpected order response: {order}")
            return order
        except Exception as exc:
            if attempt < max_attempts - 1:
                logger.warning(
                    "Retry open %s attempt %d error: %s",
                    symbol,
                    attempt + 1,
                    exc,
                )
                time.sleep(2 ** attempt)
            else:
                raise OrderSubmitError(f"Failed to open position {symbol}") from exc

    raise OrderSubmitError(f"Failed to open position {symbol}")


@circuit_breaker(fallback=None, max_failures=3, reset_timeout=60)
def close_position(symbol: str, side: str, amount: float, order_type: str = "market"):
    """Close an existing position with validation and retries."""
    if exchange is None:
        raise OrderSubmitError("Exchange not initialized")
    bitget_sym = symbol.replace("_", "/") + ":USDT"

    if not (config.TEST_MODE or isinstance(exchange, MockExchange)):
        if bitget_sym not in exchange.markets:
            raise OrderSubmitError(f"Market {bitget_sym} not available")
    max_attempts = config.ORDER_SUBMIT_ATTEMPTS
    for attempt in range(max_attempts):
        try:
            params = {"reduceOnly": True}
            ord_type = order_type.lower()
            ord_price = None
            if ord_type not in ("market", "limit"):
                ord_type = "market"
            order = exchange.create_order(
                symbol=bitget_sym,
                type=ord_type,
                side="buy" if side == "close_short" else "sell",
                amount=amount,
                price=ord_price,
                params=params,
            )
            if not isinstance(order, dict) or not {"id", "status", "average"}.issubset(order.keys()):
                raise OrderSubmitError(f"Unexpected order response: {order}")
            return order

        except Exception as exc:
            if attempt < max_attempts - 1:
                logger.warning("Retry close %s attempt %d error: %s", symbol, attempt + 1, exc)
                time.sleep(2 ** attempt)
            else:
                raise OrderSubmitError(f"Failed to close position {symbol}") from exc

    raise OrderSubmitError(f"Failed to close position {symbol}")


def cancel_order(order_id: str, symbol: str):
    """Cancel an open order if it hasn't been filled or closed."""
    bitget_sym = symbol.replace("_", "/") + ":USDT"
    if exchange is None:
        logger.error("Exchange not initialized")
        return
    try:
        info = exchange.fetch_order(order_id, bitget_sym)
        if info.get("status") != "open":
            logger.info("Skip cancel: %s already %s", order_id, info.get("status"))
            return
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


# ---------------------- Async wrappers ----------------------
async def fetch_balance_async() -> float:
    """Asynchronous wrapper for :func:`fetch_balance` with concurrency limit."""
    async with api_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fetch_balance)


async def open_position_async(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    order_type: str = "limit",
    stop_price: float | None = None,
) -> dict | None:
    """Async helper that calls :func:`open_position` under the semaphore."""
    async with api_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: open_position(symbol, side, amount, price, order_type, stop_price),
        )


async def close_position_async(
    symbol: str,
    side: str,
    amount: float,
    order_type: str = "market",
) -> dict | None:
    """Async helper for :func:`close_position` with concurrency limit."""
    async with api_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: close_position(symbol, side, amount, order_type),
        )
