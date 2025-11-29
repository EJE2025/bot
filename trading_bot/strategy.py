import logging
import os
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import time
import requests
from . import liquidity_ws
from .indicators_talib import (
    compute_rsi,
    compute_macd,
    calculate_atr,
    calculate_support_resistance,
)
from . import config, data, execution, idempotency, predictive_model, rl_agent
from . import model_seq_loader
from .latency import measure_latency


logger = logging.getLogger(__name__)


_MODEL_WEIGHT_OVERRIDE: Optional[float] = None
_NOTIFIED_NO_MODEL = False
_SEQ_MODEL = None
_SEQ_MODEL_MTIME: float | None = None
_SEQ_MODEL_PATH: str | None = None


def smooth_series(values: Sequence[float]) -> np.ndarray:
    """Return a smoothed numpy array based on configuration."""

    method = (getattr(config, "NOISE_FILTER_METHOD", "ema") or "ema").lower()
    span = max(getattr(config, "NOISE_FILTER_SPAN", 1), 1)
    series = pd.Series(list(values), dtype=float)

    if method in {"off", "none"} or span <= 1:
        return series.to_numpy()
    if method == "median":
        smoothed = series.rolling(window=span, min_periods=1).median()
    elif method == "sma":
        smoothed = series.rolling(window=span, min_periods=1).mean()
    else:
        smoothed = series.ewm(span=span, adjust=False).mean()
    return smoothed.to_numpy()


def _is_dummy_model(model: object) -> bool:
    return bool(getattr(model, "_is_dummy_model", False))


def set_model_weight_override(weight: Optional[float]) -> None:
    """Override the configured model weight at runtime."""
    global _MODEL_WEIGHT_OVERRIDE
    if weight is None:
        _MODEL_WEIGHT_OVERRIDE = None
        return
    _MODEL_WEIGHT_OVERRIDE = max(0.0, min(1.0, float(weight)))


def get_model_weight() -> float:
    """Return the active model weight considering runtime overrides."""
    if _MODEL_WEIGHT_OVERRIDE is not None:
        return _MODEL_WEIGHT_OVERRIDE
    return config.MODEL_WEIGHT


def get_seq_model_weight() -> float:
    """Weight assigned to the sequential model when blending probabilities."""

    return max(0.0, min(1.0, getattr(config, "MODEL_SEQ_WEIGHT", 0.0)))


def blend_probabilities(
    orig_prob: float,
    model_prob: float | None,
    weight: float | None = None,
    *,
    seq_prob: float | None = None,
    seq_weight: float | None = None,
) -> float:
    """Combine heuristic and model probabilities respecting ``weight``."""

    blended = max(0.0, min(1.0, orig_prob))
    if model_prob is not None:
        active_weight = get_model_weight() if weight is None else max(0.0, min(1.0, weight))
        blended = active_weight * model_prob + (1.0 - active_weight) * blended
    if seq_prob is not None:
        active_seq_weight = (
            get_seq_model_weight() if seq_weight is None else max(0.0, min(1.0, seq_weight))
        )
        blended = active_seq_weight * seq_prob + (1.0 - active_seq_weight) * blended
    return max(0.0, min(1.0, blended))


def _load_seq_model_if_needed():
    global _SEQ_MODEL, _SEQ_MODEL_MTIME, _SEQ_MODEL_PATH

    seq_path = getattr(config, "MODEL_SEQ_PATH", "").strip()
    if not seq_path:
        return None

    try:
        mtime = os.path.getmtime(seq_path)
    except OSError:
        if _SEQ_MODEL is None:
            logger.debug("Sequential model path %s not accessible", seq_path)
        _SEQ_MODEL = None
        _SEQ_MODEL_MTIME = None
        _SEQ_MODEL_PATH = None
        return None

    if _SEQ_MODEL is None or _SEQ_MODEL_PATH != seq_path or (
        _SEQ_MODEL_MTIME is None or mtime > _SEQ_MODEL_MTIME
    ):
        try:
            _SEQ_MODEL = model_seq_loader.load_sequence_model(seq_path)
            _SEQ_MODEL_PATH = seq_path
            _SEQ_MODEL_MTIME = mtime
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load sequential model from %s: %s", seq_path, exc)
            _SEQ_MODEL = None
            return None
    return _SEQ_MODEL


def _normalize_sequence_features(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True)
    return (sequence - mean) / (std + 1e-8)


def _build_sequence_tensor(symbol: str, window: int, interval: str) -> np.ndarray | None:
    if window <= 0:
        return None
    market = data.get_market_data(symbol, interval=interval, limit=window)
    if not market:
        return None

    try:
        volume_series = market.get("vol") or market.get("volume")
        stacked = np.column_stack(
            [market["close"], market["high"], market["low"], volume_series]
        )
    except Exception:
        logger.debug("[%s] Incomplete market data for sequential model", symbol)
        return None

    if stacked.shape[0] < window:
        logger.debug("[%s] Not enough candles for sequential model (got %d)", symbol, stacked.shape[0])
        return None

    windowed = np.asarray(stacked[-window:], dtype=float)
    normalized = _normalize_sequence_features(windowed)
    return np.expand_dims(normalized, axis=0)


def passes_probability_threshold(
    prob: float,
    risk_reward: float,
    volatility: float | None = None,
    **kwargs,
) -> bool:
    threshold = probability_threshold(
        risk_reward, volatility=volatility, **kwargs
    )
    return prob >= threshold


def _round_trip_cost_fraction(
    entry_price: float | None,
    *,
    bids_top: Sequence[Sequence[float]] | None = None,
    asks_top: Sequence[Sequence[float]] | None = None,
) -> float:
    """Estimate round-trip trading costs as a fraction of notional."""

    fee_in = getattr(config, "TAKER_FEE_RATE", config.FEE_EST)
    fee_out = getattr(config, "TAKER_FEE_RATE", config.FEE_EST)
    spread_fraction = 0.0

    if entry_price and entry_price > 0 and bids_top and asks_top:
        try:
            top_bid = float(bids_top[0][0])
            top_ask = float(asks_top[0][0])
            spread = max(0.0, top_ask - top_bid)
            reference_price = max((top_bid + top_ask) / 2.0, entry_price, 1e-9)
            spread_fraction = spread / reference_price
        except Exception:  # pragma: no cover - defensive
            spread_fraction = 0.0

    return max(0.0, fee_in + fee_out + spread_fraction)


def _breakeven_probability(risk_reward: float, cost_fraction: float) -> float:
    if risk_reward <= 0:
        return 1.0
    cost = max(cost_fraction, 0.0)
    return cost / max(risk_reward + cost, 1e-9)


def probability_threshold(
    risk_reward: float,
    *,
    volatility: float | None = None,
    cost_fraction: float | None = None,
    entry_price: float | None = None,
    bids_top: Sequence[Sequence[float]] | None = None,
    asks_top: Sequence[Sequence[float]] | None = None,
) -> float:
    """Compute the minimum probability required to keep a signal."""
    if risk_reward <= 0:
        return 1.0

    configured_threshold = getattr(config, "PROB_THRESHOLD", None)
    base_threshold = config.MIN_PROB_SUCCESS
    base_with_margin = base_threshold
    if configured_threshold is not None:
        base_threshold = max(base_threshold, configured_threshold)
        if configured_threshold > config.MIN_PROB_SUCCESS:
            base_with_margin = base_threshold + getattr(config, "FEE_AWARE_MARGIN_BPS", 0) / 10000.0
        else:
            base_with_margin = base_threshold
    fee_margin = getattr(config, "FEE_AWARE_MARGIN_BPS", 0) / 10000.0

    if cost_fraction is None:
        if entry_price is not None and entry_price > 0:
            cost_fraction = _round_trip_cost_fraction(
                entry_price, bids_top=bids_top, asks_top=asks_top
            )
        else:
            cost_fraction = config.FEE_EST

    breakeven = _breakeven_probability(risk_reward, cost_fraction)
    dynamic_threshold = breakeven + config.PROBABILITY_MARGIN + fee_margin

    if volatility is not None:
        vol_threshold = max(getattr(config, "VOL_HIGH_TH", 0.0), 0.0)
        extra_margin_bps = getattr(config, "VOL_MARGIN_BPS", 0.0)
        if volatility >= vol_threshold and extra_margin_bps > 0:
            dynamic_threshold += extra_margin_bps / 10000.0

    threshold = max(base_with_margin, dynamic_threshold)
    return min(threshold, 0.995)


def log_signal_details(
    symbol: str,
    side: str,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    prob_success: float,
    modelo_historico,
    *,
    orig_prob: float | None = None,
    model_prob: float | None = None,
    seq_prob: float | None = None,
    blended_prob: float | None = None,
    prob_threshold: float | None = None,
    breakeven_win_rate: float | None = None,
) -> None:
    """Log detailed signal information and model status."""
    global _NOTIFIED_NO_MODEL
    if modelo_historico is None:
        if not _NOTIFIED_NO_MODEL:
            logger.info(
                "Modelo histórico no cargado; se usará probabilidad heurística sin ajuste"
            )
            _NOTIFIED_NO_MODEL = True
    else:
        if _NOTIFIED_NO_MODEL:
            _NOTIFIED_NO_MODEL = False
        logger.debug("Modelo histórico cargado correctamente")

    logger.debug(
        "[%s] %s entry %.4f TP %.4f SL %.4f | Probabilidad de éxito: %.2f%%",
        symbol,
        side,
        entry_price,
        take_profit,
        stop_loss,
        prob_success * 100,
    )
    if orig_prob is not None:
        logger.debug("[%s] Probabilidad heurística: %.2f%%", symbol, orig_prob * 100)
    if model_prob is not None:
        logger.debug("[%s] Probabilidad modelo: %.2f%%", symbol, model_prob * 100)
    if seq_prob is not None:
        logger.debug("[%s] Probabilidad modelo secuencial: %.2f%%", symbol, seq_prob * 100)
    if blended_prob is not None:
        logger.debug("[%s] Probabilidad combinada: %.2f%%", symbol, blended_prob * 100)
    if prob_threshold is not None:
        logger.debug(
            "[%s] Umbral mínimo de probabilidad aplicado: %.2f%%",
            symbol,
            prob_threshold * 100,
        )
    if breakeven_win_rate is not None:
        logger.debug(
            "[%s] Win rate breakeven necesario: %.2f%%",
            symbol,
            breakeven_win_rate * 100,
        )


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


def calcular_tamano_posicion(
    balance_usdt: float,
    entry_price: float,
    atr_value: float,
    atr_multiplier: float,
    risk_per_trade_usd: float,
) -> float | None:
    """Dimensiona la posición para que la pérdida máxima sea
    ``risk_per_trade_usd``.

    Devuelve ``None`` cuando los parámetros son inválidos (ATR o precio no
    positivos, o distancia al ``stop`` no positiva) o si la cantidad resultante
    es inferior a :data:`config.MIN_POSITION_SIZE`.
    """

    # Validaciones básicas de entrada
    if atr_value <= 0 or entry_price <= 0:
        return None

    distancia_stop = atr_value * atr_multiplier
    # Si la distancia al stop es nula o negativa no se abre la operación
    if distancia_stop <= 0:
        return None

    # Cantidad de contratos acorde al riesgo permitido
    qty = risk_per_trade_usd / (distancia_stop * entry_price)

    # No superar el saldo disponible
    max_qty = balance_usdt / entry_price
    qty = max(0.0, min(qty, max_qty))

    if qty < config.MIN_POSITION_SIZE:
        if atr_value < entry_price * 0.0001:
            return config.MIN_POSITION_SIZE
        return None

    notional = qty * entry_price
    if notional < getattr(config, "MIN_POSITION_SIZE_USDT", 0.0):
        return None

    return qty


def position_sizer(symbol: str, features: dict, ctx: dict | None = None) -> float:
    """Return the desired position notional in USDT for ``symbol``.

    When :data:`config.USE_FIXED_POSITION_SIZE` is enabled the function
    short-circuits and returns :data:`config.FIXED_POSITION_SIZE_USDT` without
    consulting balances or exchange limits. In ``shadow`` mode the value is
    clamped to :data:`config.MIN_POSITION_SIZE_USDT` to keep paper trades above
    the configured minimum notional.
    """

    ctx = ctx or {}
    entry_price = float(features.get("entry_price", 0.0) or 0.0)
    atr_value = float(features.get("atr", 0.0) or 0.0)
    atr_multiplier = float(
        features.get("atr_multiplier", config.STOP_ATR_MULT)
    )

    if entry_price <= 0:
        return 0.0

    if config.USE_FIXED_POSITION_SIZE:
        fixed_size = float(config.FIXED_POSITION_SIZE_USDT)
        lower = float(getattr(config, "MIN_POSITION_SIZE_USDT", 0.0) or 0.0)
        upper = float(getattr(config, "MAX_POSITION_SIZE_USDT", 0.0) or 0.0)
        clamped = max(lower, fixed_size)
        if upper > 0:
            clamped = min(clamped, upper)
        return clamped

    balance = ctx.get("balance")
    if balance is None:
        balance = execution.fetch_balance()

    risk_usd = ctx.get("risk_usd")
    if risk_usd is None:
        if config.RISK_PER_TRADE < 1:
            risk_usd = balance * config.RISK_PER_TRADE
        else:
            risk_usd = config.RISK_PER_TRADE

    leverage = config.DEFAULT_LEVERAGE
    notional = risk_usd * leverage
    lower = float(getattr(config, "MIN_POSITION_SIZE_USDT", 0.0) or 0.0)
    upper = float(getattr(config, "MAX_POSITION_SIZE_USDT", 0.0) or 0.0)
    if lower > 0:
        notional = max(notional, lower)
    if upper > 0:
        notional = min(notional, upper)
    return notional


def risk_reward_ratio(
    entry_price: float,
    take_profit: float,
    stop_loss: float,
) -> float:
    """Return reward-to-risk ratio or ``0.0`` if risk is zero."""
    risk = abs(entry_price - stop_loss)
    if risk <= 0:
        return 0.0
    reward = abs(take_profit - entry_price)
    return reward / risk


def decidir_entrada(
    symbol: str,
    modelo_historico=None,
    info: dict | None = None,
):
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
    smoothed_closes = smooth_series(closes)
    smoothed_highs = smooth_series(highs)
    smoothed_lows = smooth_series(lows)

    rsi_arr = compute_rsi(smoothed_closes.tolist(), config.RSI_PERIOD)
    if rsi_arr.size == 0 or np.isnan(rsi_arr[-1]):
        logger.error("[%s] invalid RSI", symbol)
        return None
    rsi_val = rsi_arr[-1]

    macd_line, _, _ = compute_macd(smoothed_closes.tolist())
    if macd_line.size == 0 or np.isnan(macd_line[-1]):
        logger.error("[%s] invalid MACD", symbol)
        return None
    macd_val = macd_line[-1]

    atr_val = calculate_atr(
        smoothed_highs.tolist(), smoothed_lows.tolist(), smoothed_closes.tolist()
    )
    if atr_val is None or np.isnan(atr_val):
        logger.error("[%s] invalid ATR", symbol)
        return None

    senti = sentiment_score(symbol)
    avg_vol = np.mean(vols[-10:])
    volume_factor = min(1, avg_vol / 1000)

    book = data.get_order_book(symbol)
    if not book:
        logger.warning(
            "No se pudo obtener order book para %s; se descarta la señal",
            symbol,
        )
        return None
    ob_imb = data.order_book_imbalance(book, entry_price)
    bids_top, asks_top = data.top_liquidity_levels(book)

    score_long = (
        max(0.0, config.RSI_OVERSOLD - rsi_val)
        + max(0.0, macd_val)
        + max(0.0, senti * 10)
    )
    score_short = (
        max(0.0, rsi_val - config.RSI_OVERBOUGHT)
        + max(0.0, -macd_val)
        + max(0.0, -senti * 10)
    )
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
    if decision == "BUY":
        max_loss_price = entry_price * (1 - config.MAX_STOP_LOSS_PCT)
        stop_loss = max(stop_loss, max_loss_price)
        stop_loss = max(stop_loss, 0.0)
    else:
        max_loss_price = entry_price * (1 + config.MAX_STOP_LOSS_PCT)
        stop_loss = min(stop_loss, max_loss_price)
    trend_strength = abs(macd_val) + abs(rsi_val - 50)
    tp_factor = 3.0 if trend_strength > 20 else 2.0
    take_profit = (
        entry_price + atr_val * tp_factor
        if decision == "BUY"
        else entry_price - atr_val * tp_factor
    )

    prob_heuristic = (
        min(
            max(score_long if decision == "BUY" else score_short, 0) / 5.0,
            0.85,
        )
        * volume_factor
    )
    prob_heuristic = min(max(prob_heuristic, 0.0), 0.99)
    leverage = config.DEFAULT_LEVERAGE
    balance = execution.fetch_balance()
    if config.RISK_PER_TRADE < 1:
        risk_usd = balance * config.RISK_PER_TRADE
    else:
        risk_usd = config.RISK_PER_TRADE

    sizing_features = {
        "entry_price": entry_price,
        "atr": atr_val,
        "atr_multiplier": atr_mult,
    }
    sizing_ctx = {"balance": balance, "risk_usd": risk_usd}
    position_size_usdt = position_sizer(symbol, sizing_features, sizing_ctx)
    if position_size_usdt <= 0:
        logger.error("[%s] position size below minimum", symbol)
        return None
    quantity = position_size_usdt / entry_price

    risk = abs(entry_price - stop_loss)
    risk_reward = risk_reward_ratio(entry_price, take_profit, stop_loss)
    volatility_ratio = atr_val / entry_price if entry_price > 0 else 0.0
    cost_fraction = _round_trip_cost_fraction(
        entry_price, bids_top=bids_top, asks_top=asks_top
    )
    breakeven_prob = _breakeven_probability(risk_reward, cost_fraction)

    side_label = "long" if decision == "BUY" else "short"
    feature_snapshot = {
        "risk_reward": risk_reward,
        "orig_prob": prob_heuristic,
        "side": side_label,
        "rsi": float(rsi_val),
        "macd": float(macd_val),
        "atr": float(atr_val),
        "sentiment": float(senti),
        "order_book_imbalance": float(ob_imb),
        "volume_factor": float(volume_factor),
        "volatility": float(volatility_ratio),
    }

    rl_info = None
    try:
        rl_info = rl_agent.adjust_targets(
            feature_snapshot, entry_price, take_profit, stop_loss
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[%s] RL agent failed, using heuristics: %s", symbol, exc)
        rl_info = None

    if rl_info:
        new_tp = rl_info.get("take_profit")
        new_sl = rl_info.get("stop_loss")
        if new_tp is not None and new_sl is not None and np.isfinite(new_tp) and np.isfinite(new_sl):
            take_profit = float(new_tp)
            stop_loss = float(new_sl)
            risk = abs(entry_price - stop_loss)
            risk_reward = risk_reward_ratio(entry_price, take_profit, stop_loss)
            feature_snapshot["risk_reward"] = risk_reward
        else:
            logger.debug("[%s] RL action invalid; keeping heuristic targets", symbol)

    signal = {
        "symbol": symbol,
        "side": decision,
        "entry_price": entry_price,
        "quantity": quantity,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "prob_success": prob_heuristic,
        "orig_prob": prob_heuristic,
        "risk_reward": risk_reward,
        "volatility": volatility_ratio,
        "breakeven_win_rate": breakeven_prob,
        "timeframe": "short_term",
        "max_duration_minutes": config.MAX_TRADE_DURATION_MINUTES,
        "open_time": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "feature_snapshot": feature_snapshot,
    }

    if rl_info:
        rl_state = rl_info.get("state")
        signal["rl_action"] = {
            "tp_multiplier": rl_info.get("tp_multiplier"),
            "sl_multiplier": rl_info.get("sl_multiplier"),
            "action_index": rl_info.get("action_index"),
        }
        if rl_state is not None:
            signal["rl_state"] = np.asarray(rl_state, dtype=float).tolist()

    modelo_activo = (
        None
        if modelo_historico is None or _is_dummy_model(modelo_historico)
        else modelo_historico
    )

    model_prob: float | None = None
    seq_prob: float | None = None
    if modelo_activo is not None and risk > 0:
        X_new = pd.DataFrame([feature_snapshot])
        try:
            with measure_latency("feature_to_prediction"):
                X_validated = predictive_model.ensure_feature_schema(
                    modelo_activo, X_new
                )
                model_prob = float(
                    predictive_model.predict_proba(modelo_activo, X_validated)[0]
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[%s] modelo_historico.predict_proba failed: %s", symbol, exc)
        else:
            if model_prob is not None:
                signal["model_prob"] = model_prob
    seq_model = _load_seq_model_if_needed()
    if seq_model is not None:
        try:
            seq_tensor = _build_sequence_tensor(
                symbol, getattr(config, "MODEL_SEQ_WINDOW", 0), getattr(config, "MODEL_SEQ_INTERVAL", "Min5")
            )
            if seq_tensor is not None:
                seq_pred = model_seq_loader.predict_sequence(seq_model, seq_tensor)
                seq_prob = float(np.asarray(seq_pred).reshape(-1)[0])
                seq_prob = max(0.0, min(1.0, seq_prob))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[%s] sequential model prediction failed: %s", symbol, exc)
        else:
            if seq_prob is not None:
                signal["seq_prob"] = seq_prob

    blended_prob = blend_probabilities(
        prob_heuristic, model_prob, seq_prob=seq_prob
    )
    signal["prob_success"] = blended_prob
    final_prob = signal["prob_success"]
    try:
        threshold = probability_threshold(
            risk_reward,
            volatility=volatility_ratio,
            cost_fraction=cost_fraction,
            entry_price=entry_price,
            bids_top=bids_top,
            asks_top=asks_top,
        )
    except TypeError:
        threshold = probability_threshold(risk_reward, volatility=volatility_ratio)
    signal["prob_threshold"] = threshold
    try:
        passes_threshold = passes_probability_threshold(
            final_prob,
            risk_reward,
            volatility_ratio,
            cost_fraction=cost_fraction,
            entry_price=entry_price,
            bids_top=bids_top,
            asks_top=asks_top,
        )
    except TypeError:
        passes_threshold = passes_probability_threshold(
            final_prob, risk_reward, volatility_ratio
        )
    if not passes_threshold:
        logger.debug(
            "[%s] señal descartada por probabilidad %.2f < %.2f",
            symbol,
            final_prob,
            threshold,
        )
        return None

    if config.SIGNAL_IDEMPOTENCY_TTL > 0:
        key = idempotency.build_idempotency_key(
            signal["symbol"],
            signal["side"],
            signal["entry_price"],
            signal["quantity"],
            bucket_seconds=int(config.SIGNAL_IDEMPOTENCY_TTL),
        )
        signal["idempotency_key"] = key
        if not idempotency.should_submit(key):
            logger.debug(
                "[%s] señal duplicada ignorada (TTL=%ss)",
                symbol,
                config.SIGNAL_IDEMPOTENCY_TTL,
            )
            return None
        idempotency.store_result(key, signal)

    log_signal_details(
        symbol,
        decision,
        entry_price,
        take_profit,
        stop_loss,
        final_prob,
        modelo_activo,
        orig_prob=prob_heuristic,
        model_prob=model_prob,
        seq_prob=seq_prob,
        blended_prob=blended_prob,
        prob_threshold=threshold,
        breakeven_win_rate=breakeven_prob,
    )
    return signal


# Liquidity monitoring helper
SYMBOLS = config.SYMBOLS or ["BTC/USDT", "ETH/USDT"]


def start_liquidity(symbols=None):
    """Start the liquidity websocket listeners when running the bot."""
    if symbols is None:
        symbols = config.SYMBOLS or None
    if symbols:
        liquidity_ws.start(symbols)
    else:
        liquidity_ws.start()


def print_liquidity():
    """Example function showing how to query liquidity data."""
    while True:
        book = liquidity_ws.get_liquidity()
        for sym, book_data in book.items():
            top_bid = next(
                iter(sorted(book_data["bids"].items(), reverse=True)), None
            )
            top_ask = next(
                iter(sorted(book_data["asks"].items())), None
            )
            if top_bid and top_ask:
                print(
                    f"{sym}: bid {top_bid[0]} ({top_bid[1]}), "
                    f"ask {top_ask[0]} ({top_ask[1]})"
                )
        time.sleep(5)


if __name__ == "__main__":
    start_liquidity()
    print_liquidity()
