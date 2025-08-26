# trade_manager.py

import threading
import json
import os
from datetime import datetime, timezone
import time
import logging
import uuid
from . import config
from .utils import normalize_symbol
from .state_machine import TradeState, StatefulTrade, InvalidStateTransition

logger = logging.getLogger(__name__)

open_trades = []
closed_trades = []
trade_history = []  # Guarda todos los cambios si quieres auditar

LOCK = threading.Lock()

# Cool-down registry for recently closed symbols
_last_closed: dict[str, float] = {}


def reset_state() -> None:
    """Clear all in-memory trade state (used in tests)."""
    with LOCK:
        open_trades.clear()
        closed_trades.clear()
        trade_history.clear()
        _last_closed.clear()


def in_cooldown(symbol: str) -> bool:
    """Return ``True`` if ``symbol`` was closed recently and is cooling
    down."""
    norm = normalize_symbol(symbol)
    ts = _last_closed.get(norm)
    if ts is None:
        return False
    return (time.time() - ts) < config.TRADE_COOLDOWN


# --- Core functions ---


def add_trade(trade):
    """Añade una nueva operación a la lista de abiertas."""
    with LOCK:
        trade["symbol"] = normalize_symbol(trade.get("symbol", ""))
        if "trade_id" not in trade:
            trade["trade_id"] = str(uuid.uuid4())
        trade.setdefault(
            "open_time",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        trade.setdefault("status", "pending")
        trade.setdefault("state", TradeState.PENDING.value)
        open_trades.append(trade)
        log_history("open", trade)
        return trade


def find_trade(symbol=None, trade_id=None):
    """Devuelve la primera operación abierta que coincida con el símbolo
    o el ID."""
    norm = normalize_symbol(symbol) if symbol else None
    with LOCK:
        for trade in open_trades:

            if norm and normalize_symbol(trade.get("symbol")) == norm:
                return trade
            if trade_id and trade.get("trade_id") == trade_id:
                return trade
    return None


def update_trade(trade_id, **kwargs):
    """Actualiza los campos de una operación abierta."""
    with LOCK:
        for trade in open_trades:
            if trade.get("trade_id") == trade_id:
                trade.update(kwargs)
                log_history("update", trade)
                return True
    return False


def set_trade_state(trade_id: str, new_state: TradeState) -> None:
    t = None
    with LOCK:
        for trade in open_trades:
            if trade.get("trade_id") == trade_id:
                t = trade
                break
        if t is None:
            raise ValueError("Trade no encontrado")
        st = StatefulTrade(trade_id=trade_id, state=TradeState(t["state"]))
        if not st.can_transition_to(new_state):
            msg = f"{st.state} → {new_state} no permitido"
            raise InvalidStateTransition(msg)
        st.transition_to(new_state)
        t["state"] = st.state.value


def close_trade(
    trade_id=None,
    symbol=None,
    reason="closed",
    exit_price=None,
    profit=None,
):
    """Cierra una operación y la mueve a cerradas, añadiendo motivo.

    Parámetros adicionales permiten registrar precio de salida y beneficio
    para que el historial en JSON tenga la misma información que el CSV de
    `history`.
    """
    trade = find_trade(symbol=symbol, trade_id=trade_id)
    if not trade:
        return None
    tid = trade.get("trade_id")
    cur = TradeState(trade["state"])
    if cur in (TradeState.OPEN, TradeState.PARTIALLY_FILLED):
        set_trade_state(tid, TradeState.CLOSING)
    set_trade_state(tid, TradeState.CLOSED)
    with LOCK:
        for i, t in enumerate(open_trades):
            if t.get("trade_id") == tid:
                trade = open_trades.pop(i)
                break
        else:
            return None
        _last_closed[normalize_symbol(trade.get("symbol"))] = time.time()
        trade["close_time"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        trade["close_reason"] = reason
        if exit_price is not None:
            trade["exit_price"] = exit_price
        if profit is not None:
            trade["profit"] = profit
        trade["status"] = "closed"
        closed_trades.append(trade)
        log_history("close", trade)
        return trade


def all_open_trades():
    with LOCK:
        return list(open_trades)


def all_closed_trades():
    with LOCK:
        return list(closed_trades)

# --- Persistence ---


def atomic_write(path: str, data) -> None:
    """Write JSON data atomically to ``path``."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.error("Error saving file %s: %s", path, exc)
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_trades(
    open_path="open_trades.json",
    closed_path="closed_trades.json",
):
    """Persist open and closed trades to disk separately."""
    with LOCK:
        open_data = list(open_trades)
        closed_data = list(closed_trades)

    try:
        atomic_write(open_path, open_data)
    except Exception:
        logger.error("Failed to save open trades")

    try:
        atomic_write(closed_path, closed_data)
    except Exception:
        logger.error("Failed to save closed trades")


def load_trades(
    open_path="open_trades.json",
    closed_path="closed_trades.json",
):
    try:
        if os.path.exists(open_path):
            with open(open_path, "r") as f:
                data = json.load(f)
            with LOCK:
                open_trades.clear()
                for t in data:
                    t.setdefault("status", "active")
                    default_state = (
                        TradeState.PENDING.value
                        if t.get("status") == "pending"
                        else TradeState.OPEN.value
                    )
                    t.setdefault("state", default_state)
                open_trades.extend(data)
        if os.path.exists(closed_path):
            with open(closed_path, "r") as f:
                data = json.load(f)
            with LOCK:
                closed_trades.clear()
                for t in data:
                    t.setdefault("status", "closed")
                    t.setdefault("state", TradeState.CLOSED.value)
                closed_trades.extend(data)
    except Exception as e:
        logger.error("Error cargando trades: %s", e)

# --- Optional: Auditing/history ---


def log_history(event_type, trade):
    """Store a snapshot of the trade change if history logging is enabled."""
    if not config.ENABLE_TRADE_HISTORY_LOG:
        return
    entry = {
        "timestamp": (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "event": event_type,
        "trade_snapshot": trade.copy(),
    }
    with LOCK:
        trade_history.append(entry)
        if len(trade_history) > config.MAX_TRADE_HISTORY_SIZE:
            trade_history.pop(0)


def get_history():
    return list(trade_history)


def export_trade_history(filepath: str):
    """Export and clear the in-memory trade history."""
    with LOCK:
        data = list(trade_history)
        trade_history.clear()
    try:
        atomic_write(filepath, data)
    except Exception:
        logger.error("Failed to export trade history")

# --- Utilities ---


def count_open_trades():
    with LOCK:
        return len(open_trades)


def count_trades_for_symbol(symbol: str) -> int:
    """Return number of open trades for ``symbol``."""
    with LOCK:
        return sum(1 for t in open_trades if t.get("symbol") == symbol)
