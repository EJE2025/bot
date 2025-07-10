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
from trading_bot.trade_manager import (
    add_trade, close_trade, find_trade, all_open_trades, load_trades, save_trades, count_open_trades
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run():
    load_trades()  # Carga operaciones abiertas al arrancar

    Thread(
        target=webapp.start_dashboard,
        args=(config.WEBAPP_HOST, config.WEBAPP_PORT),
        daemon=True,
    ).start()

    strategy.start_liquidity()

    model = optimizer.load_model(config.MODEL_PATH)
    daily_profit = 0.0

    logger.info("Starting trading loop...")

    while True:
        try:
            if daily_profit < config.DAILY_RISK_LIMIT:
                logger.error("Daily limit reached %.2f", daily_profit)
                break

            # ABRIR NUEVAS OPERACIONES (solo si hay hueco)
            if count_open_trades() < config.MAX_OPEN_TRADES:
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
                        sig["order_id"] = order.get("id") if isinstance(order, dict) else None
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
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error("Loop error: %s", exc)
            time.sleep(10)

if __name__ == "__main__":
    run()

