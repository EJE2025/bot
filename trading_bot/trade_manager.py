# trade_manager.py

import threading
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import time
import logging
import uuid
from typing import Optional

from . import config, data
from .utils import normalize_symbol
from .state_machine import TradeState, is_valid_transition

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
        trade.setdefault("requested_quantity", trade.get("quantity"))
        trade.setdefault("original_quantity", trade.get("quantity"))
        trade.setdefault(
            "open_time",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        trade.setdefault("leverage", config.DEFAULT_LEVERAGE)
        trade.setdefault("status", "pending")
        trade.setdefault("state", TradeState.PENDING.value)
        trade.setdefault("timeframe", "short_term")
        trade.setdefault("max_duration_minutes", config.MAX_TRADE_DURATION_MINUTES)
        trade.setdefault("created_ts", time.time())
        trade.setdefault("quantity_remaining", trade.get("quantity"))
        trade.setdefault("realized_pnl", 0.0)
        trade.setdefault("closing", False)
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


def get_open_trade(trade_id: str) -> Optional[dict]:
    """Return a reference to the open trade with the given ``trade_id``."""
    with LOCK:
        for trade in open_trades:
            if trade.get("trade_id") == trade_id:
                return trade
    return None


def get_closed_trade(trade_id: str) -> Optional[dict]:
    with LOCK:
        for trade in closed_trades:
            if trade.get("trade_id") == trade_id:
                return trade
    return None


def _resolve_exit_price(trade: dict, exit_price: float | None) -> float:
    if exit_price is not None:
        try:
            return float(exit_price)
        except (TypeError, ValueError):
            logger.debug("Exit price inválido para trade %s", trade.get("trade_id"))
    symbol = trade.get("symbol")
    market_price = data.get_current_price_ticker(symbol)
    if market_price:
        try:
            return float(market_price)
        except (TypeError, ValueError):
            logger.debug("Precio de mercado inválido para %s", symbol)
    try:
        return float(trade.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _trade_leverage(trade: dict) -> float:
    try:
        lev = float(trade.get("leverage") or 1.0)
    except (TypeError, ValueError):
        lev = 1.0
    if lev <= 0:
        return 1.0
    return lev


def _calculate_realized_pnl(trade: dict, quantity: float, exit_price: float) -> float:
    try:
        entry_price = float(trade.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        entry_price = 0.0
    try:
        qty = float(quantity or 0.0)
    except (TypeError, ValueError):
        qty = 0.0
    qty = abs(qty)
    leverage = _trade_leverage(trade)
    if entry_price <= 0 or qty <= 0:
        return 0.0
    invested = abs(entry_price * qty) / leverage
    side = str(trade.get("side", "buy")).lower()
    if side == "buy":
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty
    pnl /= leverage
    if invested > 0:
        pnl = max(pnl, -invested)
    return pnl


def _ensure_realized_pnl(trade: dict) -> float:
    try:
        return float(trade.get("realized_pnl", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _original_quantity(trade: dict) -> float:
    quantity_keys = (
        "requested_quantity",
        "original_quantity",
        "initial_quantity",
        "orig_qty",
        "quantity_requested",
        "base_quantity",
    )
    for key in quantity_keys:
        raw = trade.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != 0:
            return abs(value)

    fallback_keys = ("quantity", "quantity_filled", "executed_quantity")
    for key in fallback_keys:
        raw = trade.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != 0:
            return abs(value)

    return 0.0


def _total_invested(trade: dict) -> float:
    try:
        entry_price = float(trade.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        entry_price = 0.0
    quantity = _original_quantity(trade)
    if entry_price <= 0 or quantity <= 0:
        return 0.0
    leverage = _trade_leverage(trade)
    return abs(entry_price * quantity) / leverage


def update_trade(trade_id, **kwargs):
    """Actualiza los campos de una operación abierta."""
    with LOCK:
        for trade in open_trades:
            if trade.get("trade_id") == trade_id:
                trade.update(kwargs)
                log_history("update", trade)
                return True
    return False


def set_trade_state(trade_id: str, new_state: TradeState) -> bool:
    t = None
    with LOCK:
        for trade in open_trades:
            if trade.get("trade_id") == trade_id:
                t = trade
                break
        if t is None:
            logger.warning("Trade %s no encontrado para cambiar a %s", trade_id, new_state)
            return False
        raw_state = t.get("state")
        try:
            current_state = TradeState(raw_state)
        except ValueError:
            current_state = TradeState.PENDING
        if current_state == new_state:
            return True
        if not is_valid_transition(current_state, new_state):
            logger.warning(
                "Transición inválida ignorada: %s → %s (trade_id=%s)",
                current_state,
                new_state,
                trade_id,
            )
            return False
        t["state"] = new_state.value
        log_history("state", t)
        logger.info("Estado %s: %s → %s", trade_id, current_state.value, new_state.value)
        return True


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
    transitioned = set_trade_state(tid, TradeState.CLOSED)
    if not transitioned:
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
        realized = _ensure_realized_pnl(trade)
        if profit is not None:
            trade["profit"] = profit
            trade["realized_pnl"] = profit
        else:
            trade.setdefault("profit", realized)
            trade["realized_pnl"] = realized
        trade["invested_value"] = _total_invested(trade)
        trade["status"] = "closed"
        trade["closing"] = False
        trade.setdefault("quantity_remaining", 0.0)
        trade["quantity"] = trade.get("quantity_remaining", 0.0)
        closed_trades.append(trade)
        log_history("close", trade)
        try:
            from trading_bot import history

            history.append_trade(trade)
        except Exception as exc:  # pragma: no cover - defensive persistence
            logger.error("Error al guardar historial CSV: %s", exc)
        return trade


def cancel_pending_trade(trade_id: str, reason: str = "pending_timeout") -> dict | None:
    """Cancel a pending trade due to timeout or websocket desync."""

    with LOCK:
        for idx, trade in enumerate(open_trades):
            if trade.get("trade_id") != trade_id:
                continue
            try:
                state = TradeState(trade.get("state"))
            except ValueError:
                state = TradeState.PENDING
            if state != TradeState.PENDING:
                logger.debug(
                    "Skip cancel %s: estado actual %s", trade_id, trade.get("state")
                )
                return None
            cancelled = open_trades.pop(idx)
            cancelled["state"] = TradeState.FAILED.value
            cancelled["status"] = "cancelled"
            cancelled["close_reason"] = reason
            cancelled["close_time"] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            cancelled.setdefault("profit", 0.0)
            cancelled.setdefault("pnl", 0.0)
            if cancelled.get("exit_price") is None:
                cancelled["exit_price"] = cancelled.get("entry_price")
            symbol_norm = normalize_symbol(cancelled.get("symbol", ""))
            _last_closed[symbol_norm] = time.time()
            closed_trades.append(cancelled)
            log_history("cancel", cancelled)
            return cancelled
    logger.debug("Trade %s no encontrado para cancelar", trade_id)
    return None


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


def close_trade_full(
    trade_id: str,
    reason: str = "manual_close",
    exit_price: float | None = None,
) -> Optional[dict]:
    """Close the full remaining quantity of a trade in an idempotent manner."""

    trade = get_open_trade(trade_id)
    if not trade:
        return get_closed_trade(trade_id)

    exit_p = exit_price
    realized_total = 0.0
    with LOCK:
        if trade.get("closing") or trade.get("status") == "closed":
            return trade.copy()
        trade["closing"] = True
        qty_remaining = trade.get("quantity_remaining")
        if qty_remaining is None:
            qty_remaining = trade.get("quantity")
        try:
            qty_remaining = float(qty_remaining or 0.0)
        except (TypeError, ValueError):
            qty_remaining = 0.0
        if qty_remaining > 0:
            exit_p = _resolve_exit_price(trade, exit_price)
            realized = _calculate_realized_pnl(trade, qty_remaining, exit_p)
            trade["realized_pnl"] = _ensure_realized_pnl(trade) + realized
        else:
            exit_p = _resolve_exit_price(trade, exit_price)
        trade["quantity_remaining"] = 0.0
        trade["quantity"] = 0.0
        realized_total = _ensure_realized_pnl(trade)
    closed = close_trade(
        trade_id=trade_id,
        reason=reason,
        exit_price=exit_p,
        profit=realized_total,
    )
    return closed or get_closed_trade(trade_id)


def close_trade_partial(
    trade_id: str,
    quantity: float,
    reason: str = "manual_partial",
    exit_price: float | None = None,
) -> Optional[dict]:
    """Partially close a trade reducing its remaining quantity."""

    if quantity <= 0:
        raise ValueError("quantity must be positive")

    exit_p = exit_price
    realized_total = 0.0
    result_snapshot: Optional[dict] = None
    with LOCK:
        trade = get_open_trade(trade_id)
        if not trade or trade.get("status") == "closed":
            closed = get_closed_trade(trade_id)
            if closed:
                return closed.copy()
            return None

        remaining = trade.get("quantity_remaining")
        if remaining is None:
            remaining = trade.get("quantity")
        try:
            remaining = float(remaining or 0.0)
        except (TypeError, ValueError):
            remaining = 0.0
        if quantity > remaining + 1e-12:
            raise ValueError("quantity exceeds remaining size")

        exit_p = _resolve_exit_price(trade, exit_price)
        pnl_delta = _calculate_realized_pnl(trade, quantity, exit_p)
        trade["realized_pnl"] = _ensure_realized_pnl(trade) + pnl_delta
        new_remaining = remaining - quantity
        if new_remaining < 1e-12:
            new_remaining = 0.0
        trade["quantity_remaining"] = new_remaining
        trade["quantity"] = new_remaining
        trade["last_partial_close"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        trade.setdefault("partial_closures", []).append(
            {
                "timestamp": trade["last_partial_close"],
                "quantity": quantity,
                "exit_price": exit_p,
                "reason": reason,
            }
        )
        if new_remaining == 0.0:
            trade["closing"] = True
        else:
            trade["closing"] = False
            trade["status"] = "open"
        log_history("partial_close", trade)
        realized_total = _ensure_realized_pnl(trade)
        result_snapshot = trade.copy()

    if new_remaining == 0.0:
        closed = close_trade(
            trade_id=trade_id,
            reason=reason,
            exit_price=exit_p,
            profit=realized_total,
        )
        return closed or get_closed_trade(trade_id)
    return result_snapshot
