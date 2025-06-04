import logging
import pandas as pd
from typing import List
from . import data, strategy

logger = logging.getLogger(__name__)


def run_backtest(symbols: List[str], interval: str = "Min15", limit: int = 500):
    results = []
    for sym in symbols:
        info = data.get_market_data(sym, interval=interval, limit=limit)
        if not info or "close" not in info:
            logger.error("No data for %s", sym)
            continue
        closes = [float(x) for x in info["close"]]
        highs = [float(x) for x in info["high"]]
        lows = [float(x) for x in info["low"]]
        vols = [float(x) for x in info["vol"]]

        profit_total = 0.0
        trades = 0
        for i in range(150, len(closes) - 10):
            window = {
                "close": closes[:i],
                "high": highs[:i],
                "low": lows[:i],
                "vol": vols[:i],
            }
            sig = strategy.decidir_entrada(sym, info=window)
            if not sig:
                continue
            entry = closes[i]
            side = sig["side"]
            sl = sig["stop_loss"]
            tp = sig["take_profit"]
            exit_price = entry
            for j in range(i + 1, min(i + 10, len(closes))):
                h = highs[j]
                l = lows[j]
                if side == "BUY":
                    if l <= sl:
                        exit_price = sl
                        break
                    if h >= tp:
                        exit_price = tp
                        break
                else:
                    if h >= sl:
                        exit_price = sl
                        break
                    if l <= tp:
                        exit_price = tp
                        break
                exit_price = closes[j]
            if side == "BUY":
                profit = exit_price - entry
            else:
                profit = entry - exit_price
            profit_total += profit
            trades += 1
        results.append({"symbol": sym, "trades": trades, "profit": profit_total})
    return pd.DataFrame(results)


if __name__ == "__main__":
    syms = ["BTC_USDT"]
    df = run_backtest(syms)
    print(df)

