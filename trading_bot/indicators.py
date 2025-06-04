"""Utility functions for technical indicators."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def ema(prices: list[float], period: int) -> np.ndarray:
    """Return exponential moving average for the given period."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period:
        return np.array([])
    k = 2.0 / (period + 1)
    ema_vals = [prices[0]]
    for price in prices[1:]:
        ema_vals.append(price * k + ema_vals[-1] * (1.0 - k))
    return np.asarray(ema_vals)


def compute_rsi(prices: list[float], period: int = 14) -> np.ndarray:
    """Return the Relative Strength Index using Wilder's smoothing."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 1:
        return np.array([])

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.divide(
        avg_gain[period:],
        avg_loss[period:],
        out=np.zeros_like(avg_gain[period:]),
        where=avg_loss[period:] != 0,
    )
    rsi = 100 - 100 / (1 + rs)

    output = np.empty_like(prices)
    output[:period] = np.nan
    output[period:] = rsi
    return output


def compute_macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    """Return MACD line, signal line and histogram."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < slow:
        return np.array([]), np.array([]), np.array([])
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    """Return the Average True Range using Wilder's smoothing."""
    try:
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period + 1:
            return None

        prev_close = closes[:-1]
        high = highs[1:]
        low = lows[1:]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

        atr_prev = np.mean(tr[:period])
        for val in tr[period:]:
            atr_prev = (atr_prev * (period - 1) + val) / period
        return float(atr_prev)
    except Exception as exc:
        logger.error("ATR calculation error: %s", exc)
        return None


def calculate_support_resistance(prices, window=20):
    if len(prices) < window:
        return None, None
    window_prices = prices[-window:]
    return min(window_prices), max(window_prices)
