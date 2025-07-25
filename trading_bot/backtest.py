import logging
import pandas as pd
from typing import List
from . import data, strategy

logger = logging.getLogger(__name__)


def run_backtest(symbols: List[str], interval: str = "Min15", limit: int = 500) -> pd.DataFrame:
    """Run a simplified historical backtest for several symbols.

    Parameters
    ----------
    symbols : List[str]
        Trading pairs in the format ``BASE_QUOTE`` (e.g. ``"BTC_USDT"``).
    interval : str, optional
        Candle interval used by :func:`data.get_market_data`. Defaults to
        ``"Min15"``.
    limit : int, optional
        Number of candles to request for each symbol. Defaults to ``500``.

    Algorithm
    ---------
    For each symbol the function downloads historical OHLCV data and iterates
    over the candles starting at index 150. At each step a window of previous
    prices is passed to :func:`strategy.decidir_entrada` in order to obtain a
    hypothetical trade signal. The trade is simulated for up to ten candles
    applying the provided take profit or stop loss. The equity curve is tracked
    to calculate the win rate, cumulative profit and the maximum drawdown.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per symbol containing the columns
        ``symbol``, ``trades``, ``profit``, ``win_rate`` and ``max_drawdown``.
    """
    results = []
    for sym in symbols:
        info = data.get_market_data(sym, interval=interval, limit=limit)
        if not isinstance(info, dict) or not all(k in info for k in ("close", "high", "low", "vol")):
            logger.error("Invalid data for %s", sym)
            continue
        closes = [float(x) for x in info["close"]]
        highs = [float(x) for x in info["high"]]
        lows = [float(x) for x in info["low"]]
        vols = [float(x) for x in info["vol"]]

        profit_total = 0.0
        trades = 0
        wins = 0
        equity = 0.0
        peak = 0.0
        dd = 0.0
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
            if profit > 0:
                wins += 1
            equity += profit
            if equity > peak:
                peak = equity
            dd = max(dd, peak - equity)
        win_rate = wins / trades if trades else 0
        results.append(
            {
                "symbol": sym,
                "trades": trades,
                "profit": profit_total,
                "win_rate": win_rate,
                "max_drawdown": dd,
            }
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a simple backtest")
    parser.add_argument("symbols", nargs="*", help="Symbols to backtest, e.g. BTC_USDT ETH_USDT")
    parser.add_argument("--interval", default="Min15", help="Candle interval")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles")
    args = parser.parse_args()

    symbols = args.symbols or ["BTC_USDT"]
    df = run_backtest(symbols, interval=args.interval, limit=args.limit)
    print(df.to_string(index=False))

