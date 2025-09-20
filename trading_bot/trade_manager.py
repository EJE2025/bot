# trade_manager.py

import threading
import json
import os
from pathlib import Path
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

# ``add_trade`` y otros métodos llaman a ``log_history`` mientras mantienen el
# cerrojo principal. Ese helper también necesita bloquear el estado global para
# evitar condiciones de carrera. Con un ``Lock`` normal esto provocaba un
# deadlock al intentar readquirirlo desde el mismo hilo cuando el historial está
# habilitado. Cambiar a ``RLock`` permite la reentrada y mantiene la seguridad
# frente a accesos concurrentes.
LOCK = threading.RLock()

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
        trade.setdefault("timeframe", "short_term")
        trade.setdefault("max_duration_minutes", config.MAX_TRADE_DURATION_MINUTES)
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
    try:
        set_trade_state(tid, TradeState.CLOSED)
    except InvalidStateTransition:
        logger.debug(
            "Forcing trade %s to CLOSED from %s", trade.get("trade_id"), cur
        )
        with LOCK:
            trade["state"] = TradeState.CLOSED.value
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


def atomic_write(path: str | os.PathLike[str], data) -> None:
    """Write JSON data atomically to ``path``."""
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, target)
    except Exception as exc:
        logger.error("Error saving file %s: %s", target, exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
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


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        logger.debug("Invalid timestamp %s", ts)
        return None


def trade_age_minutes(trade: dict, now: datetime | None = None) -> float | None:
    """Return the age of a trade in minutes or ``None`` if unknown."""
    opened = _parse_timestamp(trade.get("open_time"))
    if opened is None:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    delta = now.astimezone(timezone.utc) - opened.astimezone(timezone.utc)
    return delta.total_seconds() / 60.0


def exceeded_max_duration(
    trade: dict, now: datetime | None = None, max_minutes: int | None = None
) -> bool:
    """Return ``True`` if the trade surpassed its allowed duration."""
    if max_minutes is None:
        max_minutes = trade.get("max_duration_minutes") or config.MAX_TRADE_DURATION_MINUTES
    if max_minutes <= 0:
        return False
    age = trade_age_minutes(trade, now=now)
    if age is None:
        return False
    return age >= max_minutes
