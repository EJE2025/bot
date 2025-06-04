import numpy as np
import logging

logger = logging.getLogger(__name__)


def ema(prices, period):
    prices = np.array(prices)
    k = 2 / (period + 1)
    ema_vals = [prices[0]]
    for price in prices[1:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return np.array(ema_vals)


def compute_rsi(prices, period=14):
    prices = np.array(prices)
    if len(prices) <= period:
        return np.array([])
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100 - 100 / (1 + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = max(-delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100 - 100 / (1 + rs)
    return rsi


def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(highs, lows, closes, period=14):
    try:
        tr = [max(h - l, abs(h - c), abs(l - c)) for h, l, c in zip(highs, lows, closes)]
        return np.mean(tr[-period:])
    except Exception as exc:
        logger.error("ATR calculation error: %s", exc)
        return None


def calculate_support_resistance(prices, window=20):
    if len(prices) < window:
        return None, None
    window_prices = prices[-window:]
    return min(window_prices), max(window_prices)
