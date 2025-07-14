import logging
import numpy as np
import pandas as pd
import time
import requests
from . import liquidity_ws
from .indicators import (
    compute_rsi,
    compute_macd,
    calculate_atr,
    calculate_support_resistance,
)
from . import config, data, execution

logger = logging.getLogger(__name__)


def sentiment_score(symbol: str, period: str = "5m") -> float:
    """Market sentiment based on long/short ratio from Bitget.

    Returns a deterministic value normalized to [-0.5, 0.5].
    """
    url = "https://api.bitget.com/api/v2/mix/market/position-long-short"
    params = {"symbol": symbol, "period": period}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("code") == "00000" and data.get("data"):
            long_ratio = float(data["data"][0]["longPositionRatio"])
            short_ratio = float(data["data"][0]["shortPositionRatio"])
            return (long_ratio - short_ratio) / 2.0
    except Exception:
        pass
    return 0.0


def calcular_tamano_posicion(balance_usdt: float, entry_price: float, atr_value: float,
                             atr_multiplier: float, risk_per_trade_usd: float) -> float:
    """Size position so max loss equals risk_per_trade_usd."""
    distancia_stop = atr_value * atr_multiplier
    if distancia_stop <= 0 or entry_price <= 0:
        return 0.0

    qty = risk_per_trade_usd / (distancia_stop * entry_price)
    max_qty = balance_usdt / entry_price
    return max(0.0, min(qty, max_qty))


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
    rsi_arr = compute_rsi(closes, config.RSI_PERIOD)
    if rsi_arr.size == 0 or np.isnan(rsi_arr[-1]):
        logger.error("[%s] invalid RSI", symbol)
        return None
    rsi_val = rsi_arr[-1]

    macd_line, _, _ = compute_macd(closes)
    if macd_line.size == 0 or np.isnan(macd_line[-1]):
        logger.error("[%s] invalid MACD", symbol)
        return None
    macd_val = macd_line[-1]

    atr_val = calculate_atr(highs, lows, closes)
    if atr_val is None or np.isnan(atr_val):
        logger.error("[%s] invalid ATR", symbol)
        return None

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
    leverage = config.DEFAULT_LEVERAGE
    balance = execution.fetch_balance()
    if config.RISK_PER_TRADE < 1:
        risk_usd = balance * config.RISK_PER_TRADE
    else:
        risk_usd = config.RISK_PER_TRADE
    quantity = calcular_tamano_posicion(
        balance,
        entry_price,
        atr_val,
        atr_mult,
        risk_usd,
    )

    risk = abs(entry_price - stop_loss)

    reward = abs(take_profit - entry_price)
    risk_reward = reward / risk if risk else 0.0
    signal = {
        "symbol": symbol,
        "side": decision,
        "entry_price": entry_price,
        "quantity": quantity,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "prob_success": prob_success,
        "risk_reward": risk_reward,
        "open_timestamp": pd.Timestamp.now(),
    }

    if modelo_historico and risk > 0:
        side_ind = 1 if decision == "BUY" else 0
        X_new = pd.DataFrame(
            [[risk_reward, prob_success, side_ind]],
            columns=["risk_reward", "orig_prob", "side"],
        )
        pred_hist = modelo_historico.predict_proba(X_new)[0, 1]
        signal["prob_success"] = (prob_success + pred_hist) / 2

    logger.info("[%s] %s entry %.4f TP %.4f SL %.4f", symbol, decision, entry_price, take_profit, stop_loss)
    return signal


# Liquidity monitoring helper
SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def start_liquidity(symbols=None):
    """Start the liquidity websocket listeners when running the bot."""
    liquidity_ws.start(symbols or SYMBOLS)

def print_liquidity():
    """Example function showing how to query liquidity data."""
    while True:
        book = liquidity_ws.get_liquidity()
        for sym, data in book.items():
            top_bid = next(iter(sorted(data["bids"].items(), reverse=True)), None)
            top_ask = next(iter(sorted(data["asks"].items())), None)
            if top_bid and top_ask:
                print(f"{sym}: bid {top_bid[0]} ({top_bid[1]}), ask {top_ask[0]} ({top_ask[1]})")
        time.sleep(5)

if __name__ == "__main__":
    start_liquidity()
    print_liquidity()

