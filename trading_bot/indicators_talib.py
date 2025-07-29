import numpy as np

try:
    import talib
    USE_TALIB = True
except Exception:  # pragma: no cover - talib optional
    from .indicators import (
        ema as fallback_ema,
        compute_rsi as fallback_rsi,
        compute_macd as fallback_macd,
        calculate_atr as fallback_atr,
        calculate_support_resistance as fallback_sr,
    )
    USE_TALIB = False
else:
    from .indicators import calculate_support_resistance as fallback_sr
    from .indicators import ema as fallback_ema
    from .indicators import compute_rsi as fallback_rsi
    from .indicators import compute_macd as fallback_macd
    from .indicators import calculate_atr as fallback_atr


def ema(prices: list[float], period: int) -> np.ndarray:
    """Exponential moving average using TA-Lib when available."""
    if USE_TALIB:
        prices_np = np.asarray(prices, dtype=float)
        if len(prices_np) < period:
            return np.array([])
        return talib.EMA(prices_np, timeperiod=period)
    return fallback_ema(prices, period)


def compute_rsi(prices: list[float], period: int = 14) -> np.ndarray:
    """Relative Strength Index via TA-Lib when available."""
    if USE_TALIB:
        prices_np = np.asarray(prices, dtype=float)
        if len(prices_np) < period + 1:
            return np.array([])
        return talib.RSI(prices_np, timeperiod=period)
    return fallback_rsi(prices, period)


def compute_macd(
    prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9
):
    """MACD line, signal line and histogram using TA-Lib when available."""
    if USE_TALIB:
        prices_np = np.asarray(prices, dtype=float)
        if len(prices_np) < slow:
            return np.array([]), np.array([]), np.array([])
        macd, sig, hist = talib.MACD(
            prices_np,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal,
        )
        return macd, sig, hist
    return fallback_macd(prices, fast, slow, signal)


def calculate_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> float | None:
    """Average True Range computed with TA-Lib when available."""
    if USE_TALIB:
        highs_np = np.asarray(highs, dtype=float)
        lows_np = np.asarray(lows, dtype=float)
        closes_np = np.asarray(closes, dtype=float)
        if len(closes_np) < period + 1:
            return None
        atr_vals = talib.ATR(highs_np, lows_np, closes_np, timeperiod=period)
        return float(atr_vals[-1])
    return fallback_atr(highs, lows, closes, period)


def calculate_support_resistance(prices, window: int = 50, tolerance: float = 0.02):
    """Support and resistance levels via fallback implementation."""
    return fallback_sr(prices, window, tolerance)

