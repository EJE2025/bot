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

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run():
    operations = []
    Thread(
        target=webapp.start_dashboard,
        args=(operations, config.WEBAPP_HOST, config.WEBAPP_PORT),
        daemon=True,
    ).start()
    model = optimizer.load_model(config.MODEL_PATH)
    symbols = data.get_common_top_symbols(execution.exchange, 15)
    active_signals = {}

    for symbol in symbols:
        sig = strategy.decidir_entrada(symbol, modelo_historico=model)
        if sig:
            active_signals[symbol] = sig

    for symbol, sig in list(active_signals.items()):
        if len(operations) >= config.MAX_OPEN_TRADES:
            logger.info("Max open trades reached")
            break
        try:
            raw = symbol.replace("_", "")
            if raw in config.BLACKLIST_SYMBOLS or raw in config.UNSUPPORTED_SYMBOLS:
                logger.info("[%s] Skipped due blacklist", symbol)
                del active_signals[symbol]
                continue
            execution.setup_leverage(raw, sig["leverage"])
            order = execution.open_position(symbol, sig["side"], sig["quantity"], sig["entry_price"], order_type="limit")
            sig["order_id"] = order.get("id") if isinstance(order, dict) else None
            operations.append(sig)
            notify.send_telegram(f"Opened {symbol} {sig['side']} @ {sig['entry_price']}")
            notify.send_discord(f"Opened {symbol} {sig['side']} @ {sig['entry_price']}")
            logger.info("Opened trade: %s", order)
        except execution.OrderSubmitError:
            pass
        except Exception as exc:
            logger.error("Error processing %s: %s", symbol, exc)
        finally:
            if symbol in active_signals:
                del active_signals[symbol]

    logger.info("Starting monitoring loop...")
    successful = 0
    unsuccessful = 0
    daily_profit = 0.0

    while True:
        try:
            if daily_profit < config.DAILY_RISK_LIMIT:
                logger.error("Daily limit reached %.2f", daily_profit)
                break
            for op in list(operations):
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
                    operations.remove(op)
                    if profit >= 0:
                        successful += 1
                    else:
                        unsuccessful += 1
                    notify.send_telegram(f"Closed {op['symbol']} PnL {profit:.2f}")
                    notify.send_discord(f"Closed {op['symbol']} PnL {profit:.2f}")
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error("Loop error: %s", exc)
            time.sleep(10)


if __name__ == "__main__":
    run()

