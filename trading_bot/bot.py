import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(env_path)

import logging
import time
from threading import Thread
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import (
    config,
    data,
    execution,
    strategy,
    webapp,
    notify,
    history,
    optimizer,
    permissions,
    trade_manager,
)
from .trade_manager import (
    add_trade,
    close_trade,
    find_trade,
    update_trade,
    all_open_trades,
    load_trades,
    save_trades,
    count_open_trades,
    count_trades_for_symbol,
    set_trade_state,
)
from .state_machine import TradeState
from .metrics import start_metrics_server, update_trade_metrics
from .monitor import monitor_system
from .utils import normalize_symbol

if not config.BITGET_API_KEY:
    print("\u26a0\ufe0f  No se carg\xf3 la API KEY. Revisa si el archivo .env existe o el gestor de secretos est\xe1 configurado.")
else:
    print("\u2705 Archivo .env cargado correctamente.")

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def open_new_trade(signal: dict):
    """Open a position and track its state via ``trade_manager``."""
    symbol = signal["symbol"]
    raw = normalize_symbol(symbol).replace("_", "")

    # Register trade in pending state first
    trade = add_trade(signal)
    try:
        execution.setup_leverage(execution.exchange, raw, signal["leverage"])
        order = execution.open_position(
            symbol,
            signal["side"],
            signal["quantity"],
            signal["entry_price"],
            order_type="limit",
        )
        if not isinstance(order, dict):
            logger.warning("Order response unexpected for %s: %s", symbol, order)
            set_trade_state(trade["trade_id"], TradeState.FAILED)
            return None

        order_id = order.get("id")
        avg_price = float(order.get("average") or signal["entry_price"])
        update_trade(
            trade["trade_id"],
            order_id=order_id,
            entry_price=avg_price,
            status="active",
        )

        status = execution.fetch_order_status(order_id, symbol)
        if status == "filled":
            set_trade_state(trade["trade_id"], TradeState.OPEN)
        elif status == "partial":
            # promote to OPEN first, then PARTIALLY_FILLED
            set_trade_state(trade["trade_id"], TradeState.OPEN)
            set_trade_state(trade["trade_id"], TradeState.PARTIALLY_FILLED)
        # else: remain in PENDING

        save_trades()
        return find_trade(trade_id=trade["trade_id"])

    except permissions.PermissionError as exc:
        logger.error("Permission denied opening %s: %s", symbol, exc)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    except execution.OrderSubmitError:
        logger.error("Order submission failed for %s", symbol)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    except Exception as exc:
        logger.error("Error processing %s: %s", symbol, exc)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    return None


def close_existing_trade(trade: dict, exit_price: float, profit: float, reason: str) -> None:
    """Close a trade enforcing state transitions."""
    trade_id = trade.get("trade_id")
    symbol = trade.get("symbol")
    qty = trade.get("quantity", 0)
    close_side = "close_short" if trade.get("side") == "SELL" else "close_long"

    set_trade_state(trade_id, TradeState.CLOSING)
    order = execution.close_position(symbol, close_side, qty, order_type="market")
    status = execution.fetch_order_status(order.get("id"), symbol)

    if status in ("filled", "partial"):
        close_trade(
            trade_id=trade_id,
            reason=reason,
            exit_price=exit_price,
            profit=profit,
        )
    else:
        set_trade_state(trade_id, TradeState.FAILED)

    save_trades()


def run_one_iteration_open(model=None):
    """Execute a single iteration of the opening logic."""
    execution.cleanup_old_orders()
    if count_open_trades() >= config.MAX_OPEN_TRADES:
        return
    symbols = data.get_common_top_symbols(execution.exchange, 15)
    candidates = []
    seen = set()
    for symbol in symbols:
        if find_trade(symbol=symbol) or trade_manager.in_cooldown(symbol):
            continue
        norm = trade_manager.normalize_symbol(symbol)
        if norm in seen:
            continue
        seen.add(norm)
        raw = symbol.replace("_", "")
        if raw in config.BLACKLIST_SYMBOLS or raw in config.UNSUPPORTED_SYMBOLS:
            continue
        sig = strategy.decidir_entrada(symbol, modelo_historico=model)
        if not sig:
            continue
        if sig.get("risk_reward", 0) < config.MIN_RISK_REWARD:
            logger.debug(
                "Skip %s: RR=%.2f < %.2f", symbol, sig.get("risk_reward", 0), config.MIN_RISK_REWARD
            )
            continue
        if sig.get("quantity", 0) < config.MIN_POSITION_SIZE:
            logger.debug(
                "Skip %s: qty=%.8f < min=%.8f",
                symbol,
                sig.get("quantity", 0),
                config.MIN_POSITION_SIZE,
            )
            continue
        candidates.append(sig)

    candidates.sort(key=lambda s: s.get("prob_success", 0), reverse=True)
    for sig in candidates:
        if count_open_trades() >= config.MAX_OPEN_TRADES:
            break
        try:
            open_new_trade(sig)
        except Exception as exc:
            logger.error("Error processing %s: %s", sig.get("symbol"), exc)

def run():
    load_trades()  # Restaurar operaciones guardadas
    permissions.audit_environment(execution.exchange)

    # Sincronizar con posiciones reales en el exchange
    positions = execution.fetch_positions()
    active_symbols = set()
    for pos in positions:
        symbol = normalize_symbol(pos.get("symbol", ""))
        qty = float(pos.get("contracts", 0))
        if qty == 0:
            continue
        active_symbols.add(symbol)
        trade = find_trade(symbol=symbol)
        if trade:
            trade.update(
                quantity=qty,
                entry_price=float(pos.get("entryPrice", trade.get("entry_price", 0))),
                leverage=int(pos.get("leverage", trade.get("leverage", 1))),
                status="active",
            )
        else:
            trade = {
                "symbol": symbol,
                "side": "BUY" if pos.get("side") == "long" else "SELL",
                "quantity": qty,
                "entry_price": float(pos.get("entryPrice", 0)),
                "stop_loss": float(pos.get("stopLossPrice", 0)) or float(pos.get("entryPrice", 0)) * 0.98,
                "take_profit": float(pos.get("takeProfitPrice", 0)) or float(pos.get("entryPrice", 0)) * 1.02,
                "leverage": int(pos.get("leverage", 1)),
                "status": "active",
            }
            add_trade(trade)

    # Eliminar operaciones locales que no existan en el exchange
    for tr in list(all_open_trades()):
        if tr["symbol"] not in active_symbols:
            close_trade(trade_id=tr.get("trade_id"), reason="sync")

    # Cancelar órdenes abiertas pendientes no registradas
    for order in execution.fetch_open_orders():
        sym = normalize_symbol(order.get("symbol", ""))
        if sym not in active_symbols:
            execution.cancel_order(order.get("id"), sym)

    # Remove any stale pending orders
    execution.cleanup_old_orders()

    # Persist state after initial synchronization
    save_trades()

    # Launch the dashboard using trade_manager as the single source of trades
    # (no operations list is passed).
    Thread(
        target=webapp.start_dashboard,
        args=(config.WEBAPP_HOST, config.WEBAPP_PORT),
        daemon=True,
    ).start()

    # Expose Prometheus metrics and start system monitor
    Thread(target=start_metrics_server, args=(8001,), daemon=True).start()
    Thread(target=monitor_system, daemon=True).start()

    strategy.start_liquidity()

    model = optimizer.load_model(config.MODEL_PATH)
    if model is None:
        logger.warning(
            "No se encontró el modelo histórico en %s; las señales se generarán sin ajuste.",
            config.MODEL_PATH,
        )
    daily_profit = 0.0
    trading_active = True

    # Initial metric update
    update_trade_metrics(count_open_trades(), len(trade_manager.all_closed_trades()))

    logger.info("Starting trading loop...")

    while True:
        try:

            execution.cleanup_old_orders()
            # Detener nuevas entradas cuando la pérdida diaria alcanza el límite
            # Se acepta que DAILY_RISK_LIMIT pueda definirse como valor negativo
            # (por ejemplo -50.0) o positivo (50.0). Siempre se compara contra
            # el valor negativo correspondiente.
            loss_limit = -abs(config.DAILY_RISK_LIMIT)
            if daily_profit <= loss_limit and trading_active:
                trading_active = False
                logger.error("Daily loss limit reached %.2f", daily_profit)


            # ABRIR NUEVAS OPERACIONES (solo si hay hueco y permitido)
            if (
                trading_active
                and count_open_trades() < config.MAX_OPEN_TRADES
                and permissions.can_open_trade(execution.exchange)
            ):
                if config.TEST_MODE and config.TEST_SYMBOLS:
                    symbols = [s.replace("/", "_").replace("-", "_")
                               for s in config.TEST_SYMBOLS]
                else:
                    symbols = data.get_common_top_symbols(execution.exchange, 15)
                candidates = []
                seen = set()
                for symbol in symbols:
                    # enforce per-symbol trade limit
                    if count_trades_for_symbol(symbol) >= config.MAX_TRADES_PER_SYMBOL:
                        continue
                    raw = normalize_symbol(symbol).replace("_", "")
                    bl = {normalize_symbol(s).replace("_", "") for s in config.BLACKLIST_SYMBOLS}
                    unsup = {normalize_symbol(s).replace("_", "") for s in config.UNSUPPORTED_SYMBOLS}
                    if raw in bl or raw in unsup:
                        continue
                    if trade_manager.in_cooldown(symbol):
                        logger.info("Cooldown activo para %s; se omite nueva entrada", symbol)
                        continue
                    sig = strategy.decidir_entrada(symbol, modelo_historico=model)
                    if not sig:
                        continue
                    if sig.get("risk_reward", 0) < config.MIN_RISK_REWARD:
                        logger.debug(
                            "Skip %s: RR=%.2f < %.2f",
                            symbol,
                            sig.get("risk_reward", 0),
                            config.MIN_RISK_REWARD,
                        )
                        continue
                    if sig.get("quantity", 0) < config.MIN_POSITION_SIZE:
                        logger.debug(
                            "Skip %s: qty=%.8f < min=%.8f",
                            symbol,
                            sig.get("quantity", 0),
                            config.MIN_POSITION_SIZE,
                        )
                        continue
                    candidates.append(sig)

                candidates.sort(key=lambda s: s.get("prob_success", 0), reverse=True)
                for sig in candidates:
                    if count_open_trades() >= config.MAX_OPEN_TRADES:
                        break
                    symbol = sig["symbol"]
                    raw = normalize_symbol(symbol).replace("_", "")
                    try:
                        trade = open_new_trade(sig)
                        if not trade:
                            continue
                        notify.send_telegram(
                            f"Opened {symbol} {trade['side']} @ {trade['entry_price']}"
                        )
                        notify.send_discord(
                            f"Opened {symbol} {trade['side']} @ {trade['entry_price']}"
                        )
                        logger.info("Opened trade: %s", trade)
                    except permissions.PermissionError as exc:
                        logger.error("Permission denied for %s: %s", symbol, exc)
                        break
                    except Exception as exc:
                        logger.error("Error processing %s: %s", symbol, exc)
            elif trading_active and not permissions.can_open_trade(execution.exchange):
                logger.debug("Skipping entries: live trading permissions not granted")

            # MONITOREAR OPERACIONES ABIERTAS
            for op in list(all_open_trades()):
                price = data.get_current_price_ticker(op["symbol"])
                if not price:
                    continue
                if op["side"] == "BUY":
                    profit = (price - op["entry_price"]) * op["quantity"]
                    close = price <= op["stop_loss"] or price >= op["take_profit"]
                    side_close = "close_long"
                else:
                    profit = (op["entry_price"] - price) * op["quantity"]
                    close = price >= op["stop_loss"] or price <= op["take_profit"]
                    side_close = "close_short"

                reason = None
                if close:
                    reason = "TP" if profit >= 0 else "SL"
                if reason is None and trade_manager.exceeded_max_duration(op):
                    close = True
                    reason = "MAX_DURATION"

                if close:
                    try:

                        order = execution.close_position(op["symbol"], side_close, op["quantity"])
                        if not isinstance(order, dict):
                            logger.warning("Close response unexpected for %s: %s", op["symbol"], order)
                            continue
                        order_id = order.get("id")
                        if not execution.check_order_filled(order_id, op["symbol"]):
                            logger.warning("Close order not filled for %s", op["symbol"])
                            execution.cancel_order(order_id, op["symbol"])
                            continue
                        exec_price = float(order.get("average") or price)
                    except execution.OrderSubmitError:
                        logger.error("Failed closing %s", op["symbol"])
                        continue
                    expected = op["take_profit"] if (
                        (op["side"] == "BUY" and price >= op["take_profit"]) or
                        (op["side"] == "SELL" and price <= op["take_profit"])
                    ) else op["stop_loss"]
                    slippage = exec_price - expected
                    logger.info(
                        "Closed %s at %.4f (target %.4f) slippage %.4f",
                        op["symbol"], exec_price, expected, slippage
                    )
                    if abs(slippage) > config.MAX_SLIPPAGE:
                        logger.warning("High slippage detected on %s: %.4f", op["symbol"], slippage)
                    close_existing_trade(
                        op,
                        exec_price,
                        profit,
                        reason or ("TP" if profit >= 0 else "SL"),
                    )
                    daily_profit += profit

                    outcome = reason or ("TP" if profit >= 0 else "SL")
                    note = (
                        f"Closed {op['symbol']} {outcome} "
                        f"PnL {profit:.2f} Slippage {slippage:.4f}"
                    )
                    notify.send_telegram(note)
                    notify.send_discord(note)

            save_trades()  # Guarda el estado periódicamente
            update_trade_metrics(count_open_trades(), len(trade_manager.all_closed_trades()))

            if not trading_active and count_open_trades() == 0:
                logger.info("All positions closed after reaching daily limit")
                save_trades()
                break
            time.sleep(60)
        except KeyboardInterrupt:

            # On manual interrupt, cancel any pending orders and persist trades
            for order in execution.fetch_open_orders():
                sym = order.get("symbol", "").replace("/", "_").replace(":USDT", "")
                execution.cancel_order(order.get("id"), sym)
            save_trades()

            break
        except Exception as exc:
            logger.error("Loop error: %s", exc)
            time.sleep(10)

if __name__ == "__main__":
    run()
