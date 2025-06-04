import logging
import numpy as np
import pandas as pd
from .indicators import compute_rsi, compute_macd, calculate_atr, calculate_support_resistance
from . import config, data

logger = logging.getLogger(__name__)


def sentiment_score(symbol: str) -> float:
    # Placeholder for sentiment analysis
    return np.random.uniform(-0.5, 0.5)


def decidir_entrada(symbol: str, modelo_historico=None, info: dict | None = None):
    if info is None:
        info = data.get_market_data(symbol, interval="Min15", limit=1000)
    if not info or "close" not in info:
        logger.error("[%s] insufficient data", symbol)
        return None

    closes = [float(x) for x in info["close"]]
    highs = [float(x) for x in info["high"]]
    lows = [float(x) for x in info["low"]]
    vols = [float(x) for x in info["vol"]]

    entry_price = closes[-1]
    support, resistance = calculate_support_resistance(closes)
    rsi_val = compute_rsi(closes, 100)[-1]
    macd_val = compute_macd(closes)[0][-1]
    atr_val = calculate_atr(highs, lows, closes) or 0
    senti = sentiment_score(symbol)
    avg_vol = np.mean(vols[-10:])
    volume_factor = min(1, avg_vol / 1000)

    book = data.get_order_book(symbol)
    ob_imb = data.order_book_imbalance(book, entry_price)
    bids_top, asks_top = data.top_liquidity_levels(book)

    score_long = max(0, 45 - rsi_val) + max(0, macd_val) + max(0, senti * 10)
    score_short = max(0, rsi_val - 55) + max(0, -macd_val) + max(0, -senti * 10)
    score_long += max(0, ob_imb)
    score_short += max(0, -ob_imb)

    if bids_top and entry_price - bids_top[0][0] < atr_val:
        score_long += 2
    if asks_top and asks_top[0][0] - entry_price < atr_val:
        score_short += 2

    if support and (entry_price - support) / entry_price < 0.02:
        score_long += 5
    if resistance and (resistance - entry_price) / entry_price < 0.02:
        score_short += 5

    decision = "BUY" if score_long >= score_short else "SELL"
    atr_mult = config.STOP_ATR_MULT
    stop_loss = (
        entry_price - atr_mult * atr_val
        if decision == "BUY"
        else entry_price + atr_mult * atr_val
    )
    trend_strength = abs(macd_val) + abs(rsi_val - 50)
    tp_factor = 3.0 if trend_strength > 20 else 2.0
    take_profit = (entry_price + atr_val * tp_factor) if decision == "BUY" else (
        entry_price - atr_val * tp_factor)

    prob_success = min(max(score_long if decision == "BUY" else score_short, 0) / 5.0, 0.85) * volume_factor
    leverage = 10
    quantity = 10 / entry_price

    signal = {
        "symbol": symbol,
        "side": decision,
        "entry_price": entry_price,
        "quantity": quantity,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "prob_success": prob_success,
        "open_timestamp": pd.Timestamp.now(),
    }

    if modelo_historico:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk > 0:
            risk_reward = reward / risk
            side_ind = 1 if decision == "BUY" else 0
            X_new = pd.DataFrame([[risk_reward, prob_success, side_ind]], columns=["risk_reward", "orig_prob", "side"])
            pred_hist = modelo_historico.predict_proba(X_new)[0, 1]
            signal["prob_success"] = (prob_success + pred_hist) / 2

    logger.info("[%s] %s entry %.4f TP %.4f SL %.4f", symbol, decision, entry_price, take_profit, stop_loss)
    return signal
