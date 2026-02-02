from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import importlib.util
import numpy as np
import pandas as pd

argrelextrema = None
if importlib.util.find_spec("scipy.signal") is not None:  # pragma: no cover - optional dependency
    from scipy.signal import argrelextrema  # type: ignore[no-redef]


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename: dict[str, str] = {}
    for col in out.columns:
        lower = col.lower()
        if lower in ("vol", "volumeto", "volume"):
            rename[col] = "volume"
    if rename:
        out = out.rename(columns=rename)
    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr.fillna(method="bfill")


def _local_extrema_idxs(series: np.ndarray, order: int = 5) -> Tuple[List[int], List[int]]:
    if len(series) < (2 * order + 1):
        return [], []
    if argrelextrema is not None:
        max_idx = argrelextrema(series, np.greater, order=order)[0].tolist()
        min_idx = argrelextrema(series, np.less, order=order)[0].tolist()
        return max_idx, min_idx

    max_idx: List[int] = []
    min_idx: List[int] = []
    for i in range(order, len(series) - order):
        window = series[i - order : i + order + 1]
        if series[i] == window.max() and series[i] > window[:order].max() and series[i] > window[order + 1 :].max():
            max_idx.append(i)
        if series[i] == window.min() and series[i] < window[:order].min() and series[i] < window[order + 1 :].min():
            min_idx.append(i)
    return max_idx, min_idx


def detect_channel(df: pd.DataFrame, lookback: int = 200) -> Optional[Dict[str, Any]]:
    w = df.tail(lookback).dropna(subset=["close"])
    y = w["close"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 30:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept
    resid = y - line
    width = float(np.std(resid) * 2.0)
    if abs(slope) < 1e-9:
        direction = "sideways"
    else:
        direction = "bullish" if slope > 0 else "bearish"
    return {
        "direction": direction,
        "slope": float(slope),
        "upper": {
            "x1": int(x[0]),
            "y1": float(line[0] + width),
            "x2": int(x[-1]),
            "y2": float(line[-1] + width),
        },
        "lower": {
            "x1": int(x[0]),
            "y1": float(line[0] - width),
            "x2": int(x[-1]),
            "y2": float(line[-1] - width),
        },
        "width": width,
        "lookback": int(lookback),
    }


def detect_support_resistance_zones(
    df: pd.DataFrame, pivot_order: int = 5, lookback: int = 300
) -> Dict[str, List[Dict[str, Any]]]:
    w = df.tail(lookback)
    atr = compute_atr(w).iloc[-1]
    tol = float(max(atr * 0.6, w["close"].iloc[-1] * 0.0015))

    highs = w["high"].to_numpy(dtype=float)
    lows = w["low"].to_numpy(dtype=float)
    max_idx, min_idx = _local_extrema_idxs(w["close"].to_numpy(dtype=float), order=pivot_order)

    levels: List[float] = []
    for i in max_idx:
        levels.append(float(highs[i]))
    for i in min_idx:
        levels.append(float(lows[i]))
    levels.sort()

    clusters: List[List[float]] = []
    for lvl in levels:
        placed = False
        for cl in clusters:
            if abs(lvl - np.mean(cl)) <= tol:
                cl.append(lvl)
                placed = True
                break
        if not placed:
            clusters.append([lvl])

    zones = []
    for cl in clusters:
        mean = float(np.mean(cl))
        touches = int(len(cl))
        zones.append(
            {
                "from": float(mean - tol / 2),
                "to": float(mean + tol / 2),
                "level": mean,
                "touches": touches,
            }
        )

    zones.sort(key=lambda z: z["touches"], reverse=True)

    price = float(w["close"].iloc[-1])
    supports = [z for z in zones if z["level"] <= price][:6]
    resist = [z for z in zones if z["level"] > price][:6]
    return {"support_zones": supports, "resistance_zones": resist}


def detect_rsi_divergences(
    df: pd.DataFrame, rsi_period: int = 14, pivot_order: int = 5, lookback: int = 300
) -> List[Dict[str, Any]]:
    w = df.tail(lookback)
    rsi = compute_rsi(w["close"], period=rsi_period).to_numpy(dtype=float)
    close = w["close"].to_numpy(dtype=float)

    peaks, troughs = _local_extrema_idxs(close, order=pivot_order)
    divs: List[Dict[str, Any]] = []

    for a, b in zip(peaks[:-1], peaks[1:]):
        if close[b] > close[a] and rsi[b] < rsi[a] - 2:
            divs.append(
                {
                    "type": "bearish",
                    "prev_idx": int(a),
                    "idx": int(b),
                    "prev_price": float(close[a]),
                    "price": float(close[b]),
                    "prev_rsi": float(rsi[a]),
                    "rsi": float(rsi[b]),
                }
            )

    for a, b in zip(troughs[:-1], troughs[1:]):
        if close[b] < close[a] and rsi[b] > rsi[a] + 2:
            divs.append(
                {
                    "type": "bullish",
                    "prev_idx": int(a),
                    "idx": int(b),
                    "prev_price": float(close[a]),
                    "price": float(close[b]),
                    "prev_rsi": float(rsi[a]),
                    "rsi": float(rsi[b]),
                }
            )

    return divs[-5:]


def detect_market_structure(
    df: pd.DataFrame, pivot_order: int = 5, lookback: int = 300
) -> List[Dict[str, Any]]:
    w = df.tail(lookback)
    close = w["close"].to_numpy(dtype=float)
    peaks, troughs = _local_extrema_idxs(close, order=pivot_order)

    events: List[Dict[str, Any]] = []
    if not peaks or not troughs:
        return events

    last_close = float(close[-1])
    last_peak = peaks[-1]
    last_trough = troughs[-1]

    prev_peak = peaks[-2] if len(peaks) >= 2 else None
    prev_trough = troughs[-2] if len(troughs) >= 2 else None

    trend = "sideways"
    if prev_peak is not None and prev_trough is not None:
        if close[last_peak] > close[prev_peak] and close[last_trough] > close[prev_trough]:
            trend = "bullish"
        elif close[last_peak] < close[prev_peak] and close[last_trough] < close[prev_trough]:
            trend = "bearish"

    if prev_peak is not None and last_close > float(close[prev_peak]):
        events.append(
            {
                "event": "BOS" if trend == "bullish" else "CHoCH",
                "direction": "bullish",
                "level": float(close[prev_peak]),
            }
        )
    if prev_trough is not None and last_close < float(close[prev_trough]):
        events.append(
            {
                "event": "BOS" if trend == "bearish" else "CHoCH",
                "direction": "bearish",
                "level": float(close[prev_trough]),
            }
        )

    return events[-3:]


def propose_trade_idea(
    df: pd.DataFrame,
    zones: Dict[str, List[Dict[str, Any]]],
    divergences: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    price = float(df["close"].iloc[-1])
    atr = float(compute_atr(df).iloc[-1])

    supports = zones.get("support_zones", [])
    resists = zones.get("resistance_zones", [])

    near_support = supports[0] if supports else None
    near_resist = resists[0] if resists else None

    last_div = divergences[-1] if divergences else None
    if not last_div:
        return None

    if last_div["type"] == "bearish" and near_resist:
        entry = price
        stop = float(near_resist["to"] + 0.5 * atr)
        tp = float(near_support["from"]) if near_support else float(price - 2.0 * atr)
        rr = abs(entry - tp) / max(abs(entry - stop), 1e-9)
        return {
            "direction": "short",
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": tp,
            "risk_reward": float(rr),
            "reasoning": "Divergencia RSI bajista cerca de resistencia.",
        }

    if last_div["type"] == "bullish" and near_support:
        entry = price
        stop = float(near_support["from"] - 0.5 * atr)
        tp = float(near_resist["to"]) if near_resist else float(price + 2.0 * atr)
        rr = abs(tp - entry) / max(abs(entry - stop), 1e-9)
        return {
            "direction": "long",
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": tp,
            "risk_reward": float(rr),
            "reasoning": "Divergencia RSI alcista cerca de soporte.",
        }

    return None


def build_technical_analysis(symbol: str, interval: str, df: pd.DataFrame) -> Dict[str, Any]:
    df = _ensure_ohlcv(df)
    channel = detect_channel(df)
    zones = detect_support_resistance_zones(df)
    divs = detect_rsi_divergences(df)
    structure = detect_market_structure(df)
    idea = propose_trade_idea(df, zones, divs)

    trend = channel["direction"] if channel else "sideways"
    return {
        "symbol": symbol,
        "interval": interval,
        "trend": {"direction": trend},
        "channel": channel,
        **zones,
        "rsi_divergences": divs,
        "market_structure": structure,
        "trade_idea": idea,
    }
