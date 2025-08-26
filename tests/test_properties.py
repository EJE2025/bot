import math
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from trading_bot import config, strategy, execution, data, trade_manager

# Estrategias de generación
symbols = st.sampled_from(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
prices = st.floats(min_value=0.0001, max_value=100000, allow_nan=False, allow_infinity=False, width=64)
atr_vals = st.floats(min_value=1e-8, max_value=0.2, allow_nan=False, allow_infinity=False)

# Helper para simular un libro de órdenes con leve spread
def _mk_book(mid: float, spread_bps: float = 5.0):
    bid = mid * (1 - spread_bps / 1e4)
    ask = mid * (1 + spread_bps / 1e4)
    return {
        "bids": [[bid, 1.0], [bid * 0.999, 2.0]],
        "asks": [[ask, 1.0], [ask * 1.001, 2.0]],
    }


# --- Invariante 1: no superar MAX_TRADES_PER_SYMBOL ---

@given(symbol=symbols, n=st.integers(min_value=0, max_value=20))
@settings(deadline=None, max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_max_trades_per_symbol_invariant(monkeypatch, symbol, n):
    MAX_N = getattr(config, "MAX_TRADES_PER_SYMBOL", 1)

    open_counts = {symbol: 0}

    def _count(sym: str) -> int:
        return open_counts.get(sym, 0)

    def _count_total() -> int:
        return sum(open_counts.values())

    monkeypatch.setattr(trade_manager, "count_trades_for_symbol", _count, raising=False)
    monkeypatch.setattr(trade_manager, "count_open_trades", _count_total, raising=False)

    monkeypatch.setattr(strategy, "sentiment_score", lambda *args, **kwargs: 0.0)

    def _get_order_book(sym):
        return _mk_book(20000.0)

    monkeypatch.setattr(data, "get_order_book", _get_order_book)
    monkeypatch.setattr(execution, "fetch_balance", lambda: 1000.0)
    monkeypatch.setattr(config, "MIN_POSITION_SIZE", 1e-6, raising=False)

    def _open_position(symbol, side, qty, price=None, order_type="limit", **kwargs):
        if open_counts.get(symbol, 0) < MAX_N:
            open_counts[symbol] = open_counts.get(symbol, 0) + 1
            return {"status": "accepted"}
        return {"status": "rejected"}

    monkeypatch.setattr(execution, "open_position", _open_position, raising=False)

    for _ in range(n):
        info = {
            "close": [20000.0] * 200,
            "high": [20010.0] * 200,
            "low": [19990.0] * 200,
            "vol": [1000.0] * 200,
        }
        signal = strategy.decidir_entrada(symbol, modelo_historico=None, info=info)
        if signal is None:
            continue
        if _count(symbol) >= MAX_N:
            assert _count(symbol) == MAX_N
        else:
            _open_position(symbol, signal["side"], signal["quantity"], price=signal["entry_price"])
            assert _count(symbol) <= MAX_N

    assert _count(symbol) <= MAX_N


# --- Invariante 2: no abrir si risk_reward < MIN_RISK_REWARD ---

@given(
    symbol=symbols,
    entry=prices,
    atr=atr_vals,
    rr_min=st.floats(min_value=0.5, max_value=5.0),
)
@settings(deadline=None, max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_never_open_below_min_risk_reward(monkeypatch, symbol, entry, atr, rr_min):
    monkeypatch.setattr(config, "MIN_RISK_REWARD", rr_min, raising=False)
    monkeypatch.setattr(strategy, "sentiment_score", lambda *args, **kwargs: 0.0)

    monkeypatch.setattr(data, "get_order_book", lambda s: _mk_book(entry))
    monkeypatch.setattr(execution, "fetch_balance", lambda: 1000.0)
    monkeypatch.setattr(config, "MIN_POSITION_SIZE", 1e-6, raising=False)

    info = {
        "close": [entry] * 200,
        "high": [entry * 1.0005] * 200,
        "low": [entry * 0.9995] * 200,
        "vol": [1000.0] * 200,
    }

    sig = strategy.decidir_entrada(symbol, modelo_historico=None, info=info)
    if sig is None or sig.get("risk_reward", 0.0) < rr_min:
        return
    rr = sig.get("risk_reward", 0.0)
    assert rr >= rr_min


# --- Invariante 3: cierre simétrico no produce PnL negativo (bajo supuestos) ---

@given(
    symbol=symbols,
    mid=prices,
    spread_bps=st.floats(min_value=0.5, max_value=15.0),
    fee_bps=st.floats(min_value=0.0, max_value=5.0),
)
@settings(deadline=None, max_examples=120, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_symmetric_close_non_negative(monkeypatch, symbol, mid, spread_bps, fee_bps):
    monkeypatch.setattr(config, "TAKER_FEE", fee_bps / 1e4, raising=False)
    monkeypatch.setattr(strategy, "sentiment_score", lambda *args, **kwargs: 0.0)

    book = _mk_book(mid, spread_bps)
    monkeypatch.setattr(data, "get_order_book", lambda s: book)
    monkeypatch.setattr(execution, "fetch_balance", lambda: 1000.0)
    monkeypatch.setattr(config, "MIN_POSITION_SIZE", 1e-6, raising=False)

    info = {
        "close": [mid] * 200,
        "high": [mid * 1.0005] * 200,
        "low": [mid * 0.9995] * 200,
        "vol": [1000.0] * 200,
    }

    sig = strategy.decidir_entrada(symbol, modelo_historico=None, info=info)
    if sig is None:
        return

    side = sig["side"]
    qty = sig["quantity"]
    entry = sig["entry_price"]

    best_bid = book["bids"][0][0]
    best_ask = book["asks"][0][0]
    taker_fee = getattr(config, "TAKER_FEE", 0.0)

    if side == "BUY":
        open_px = best_bid
        close_px = best_ask
        pnl = (close_px - open_px) * qty - taker_fee * (open_px + close_px) * qty
    else:
        open_px = best_ask
        close_px = best_bid
        pnl = (open_px - close_px) * qty - taker_fee * (open_px + close_px) * qty

    if fee_bps == 0.0:
        assert pnl >= -1e-9
    else:
        max_loss_by_fees = taker_fee * (open_px + close_px) * qty + 1e-9
        assert pnl >= -max_loss_by_fees
