import logging
import time
from threading import Thread

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
)
from .trade_manager import (
    add_trade,
    close_trade,
    find_trade,
    all_open_trades,
    load_trades,
    save_trades,
    count_open_trades,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run():
    load_trades()  # Carga operaciones abiertas al arrancar

    # Reconcile with actual exchange positions
    for pos in execution.fetch_positions():
        symbol = pos.get("symbol", "").replace("/", "_").replace(":USDT", "")
        if find_trade(symbol=symbol):
            continue
        trade = {
            "symbol": symbol,
            "side": "BUY" if pos.get("side") == "long" else "SELL",
            "quantity": float(pos.get("contracts", 0)),
            "entry_price": float(pos.get("entryPrice", 0)),
            "stop_loss": float(pos.get("stopLossPrice", 0)) or float(pos.get("entryPrice", 0)) * 0.98,
            "take_profit": float(pos.get("takeProfitPrice", 0)) or float(pos.get("entryPrice", 0)) * 1.02,
            "leverage": int(pos.get("leverage", 1)),
            "status": "active",
        }
        add_trade(trade)

    Thread(
        target=webapp.start_dashboard,
        args=(config.WEBAPP_HOST, config.WEBAPP_PORT),
        daemon=True,
    ).start()

    strategy.start_liquidity()

    model = optimizer.load_model(config.MODEL_PATH)
    daily_profit = 0.0
    trading_active = True

    logger.info("Starting trading loop...")

    while True:
        try:
            if daily_profit <= config.DAILY_RISK_LIMIT and trading_active:
                trading_active = False
                logger.error("Daily loss limit reached %.2f", daily_profit)


            # ABRIR NUEVAS OPERACIONES (solo si hay hueco y permitido)
            if trading_active and count_open_trades() < config.MAX_OPEN_TRADES:
                symbols = data.get_common_top_symbols(execution.exchange, 15)
                candidates = []
                for symbol in symbols:
                    if find_trade(symbol=symbol):
                        continue
                    raw = symbol.replace("_", "")
                    if raw in config.BLACKLIST_SYMBOLS or raw in config.UNSUPPORTED_SYMBOLS:
                        continue
                    sig = strategy.decidir_entrada(symbol, modelo_historico=model)
                    if not sig or sig.get("risk_reward", 0) < 2.0:
                        continue
                    candidates.append(sig)

                candidates.sort(key=lambda s: s.get("prob_success", 0), reverse=True)
                for sig in candidates:
                    if count_open_trades() >= config.MAX_OPEN_TRADES:
                        break
                    symbol = sig["symbol"]
                    raw = symbol.replace("_", "")
                    try:
                        execution.setup_leverage(raw, sig["leverage"])
                        order = execution.open_position(
                            symbol,
                            sig["side"],
                            sig["quantity"],
                            sig["entry_price"],
                            order_type="limit",
                        )
                        if not isinstance(order, dict):
                            logger.warning("Order response unexpected for %s: %s", symbol, order)
                            continue
                        order_id = order.get("id")
                        if not execution.check_order_filled(order_id, symbol):
                            logger.warning("Order not filled for %s", symbol)
                            execution.cancel_order(order_id, symbol)
                            continue
                        avg_price = float(order.get("average") or sig["entry_price"])
                        slippage = abs(avg_price - sig["entry_price"]) / sig["entry_price"]
                        if slippage > 0.01:
                            logger.warning("High slippage %.2f%% on %s", slippage*100, symbol)
                        sig["entry_price"] = avg_price
                        sig["order_id"] = order_id
                        sig["status"] = "active"
                        add_trade(sig)
                        notify.send_telegram(
                            f"Opened {symbol} {sig['side']} @ {sig['entry_price']}"
                        )
                        notify.send_discord(
                            f"Opened {symbol} {sig['side']} @ {sig['entry_price']}"
                        )
                        logger.info("Opened trade: %s", order)
                    except execution.OrderSubmitError:
                        logger.error("Order submission failed for %s", symbol)
                    except Exception as exc:
                        logger.error("Error processing %s: %s", symbol, exc)

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

                if close:
                    try:
                        execution.close_position(op["symbol"], side_close, op["quantity"])
                    except execution.OrderSubmitError:
                        logger.error("Failed closing %s", op["symbol"])
                        continue
                    op["close_timestamp"] = pd.Timestamp.now()
                    op["exit_price"] = price
                    op["profit"] = profit
                    history.append_trade(op)
                    daily_profit += profit
                    close_trade(trade_id=op.get("trade_id"), reason="TP" if profit >= 0 else "SL")
                    notify.send_telegram(f"Closed {op['symbol']} PnL {profit:.2f}")
                    notify.send_discord(f"Closed {op['symbol']} PnL {profit:.2f}")

            save_trades()  # Guarda el estado periódicamente
            if not trading_active and count_open_trades() == 0:
                logger.info("All positions closed after reaching daily limit")
                break
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error("Loop error: %s", exc)
            time.sleep(10)

if __name__ == "__main__":
    run()
