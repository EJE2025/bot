# trade_manager.py

import threading
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import time
import logging
import uuid
from typing import Any, Optional, Callable

from . import config, data, rl_agent
from .utils import normalize_symbol
from .state_machine import TradeState, is_valid_transition

logger = logging.getLogger(__name__)


def _trace(message: str, *args: Any) -> None:
    if config.DEBUG_TRACE:
        logger.debug(message, *args)

POSITION_RECONCILE_INTERVAL = 10
BALANCE_MONITOR_INTERVAL = 60

_reconcile_thread: threading.Thread | None = None
_balance_thread: threading.Thread | None = None

open_trades = []
closed_trades = []
trade_history = []  # Guarda todos los cambios si quieres auditar
_balance_snapshot: dict[str, float] = {
    "free_usdt": 0.0,
    "timestamp": 0.0,
    "daily_start_balance": 0.0,
    "daily_min_balance": 0.0,
    "daily_drawdown_pct": 0.0,
    "date": "",
}

# ``add_trade`` y otros métodos llaman a ``log_history`` mientras mantienen el
# cerrojo principal. Ese helper también necesita bloquear el estado global para
# evitar condiciones de carrera. Con un ``Lock`` normal esto provocaba un
# deadlock al intentar readquirirlo desde el mismo hilo cuando el historial está
# habilitado. Cambiar a ``RLock`` permite la reentrada y mantiene la seguridad
# frente a accesos concurrentes.
LOCK = threading.RLock()

# Cool-down registry for recently closed symbols
_last_closed: dict[str, float] = {}
_profit_observers: list[Callable[[float, dict], None]] = []


def reset_state() -> None:
    """Clear all in-memory trade state (used in tests)."""
    with LOCK:
        open_trades.clear()
        closed_trades.clear()
        trade_history.clear()
        _last_closed.clear()
        _profit_observers.clear()
        _balance_snapshot.update(
            {
                "free_usdt": 0.0,
                "timestamp": 0.0,
                "daily_start_balance": 0.0,
                "daily_min_balance": 0.0,
                "daily_drawdown_pct": 0.0,
                "date": "",
            }
        )
    _trace("reset_state completed")


def _prune_closed_trades() -> None:
    """Trim closed trades to avoid unbounded memory growth."""
    if config.MAX_CLOSED_TRADES <= 0:
        return
    overflow = len(closed_trades) - config.MAX_CLOSED_TRADES
    if overflow > 0:
        del closed_trades[0:overflow]


def register_profit_observer(callback: Callable[[float, dict], None]) -> None:
    """Register a callback invoked whenever a trade is closed.

    The callback receives the realized profit (or loss) and a snapshot of the
    closed trade.
    """

    with LOCK:
        if callback in _profit_observers:
            return
        _profit_observers.append(callback)


def _notify_profit_observers(profit: float, trade: dict) -> None:
    observers: list[Callable[[float, dict], None]]
    with LOCK:
        observers = list(_profit_observers)

    for cb in observers:
        try:
            cb(profit, dict(trade))
        except Exception:  # pragma: no cover - defensive observer execution
            logger.debug("Profit observer failed", exc_info=True)


def _record_balance_snapshot(amount: float) -> None:
    today = datetime.now(timezone.utc).date().isoformat()

    with LOCK:
        if _balance_snapshot.get("date") != today or _balance_snapshot.get("daily_start_balance", 0.0) <= 0:
            _balance_snapshot["daily_start_balance"] = amount
            _balance_snapshot["daily_min_balance"] = amount
            _balance_snapshot["date"] = today
        else:
            _balance_snapshot["daily_min_balance"] = min(
                float(_balance_snapshot.get("daily_min_balance") or 0.0), amount
            )

        start_balance = float(_balance_snapshot.get("daily_start_balance") or 0.0)
        min_balance = float(_balance_snapshot.get("daily_min_balance") or 0.0)
        drawdown_pct = 0.0
        if start_balance > 0:
            drawdown_pct = ((min_balance - start_balance) / start_balance) * 100

        _balance_snapshot["free_usdt"] = amount
        _balance_snapshot["timestamp"] = time.time()
        _balance_snapshot["daily_drawdown_pct"] = drawdown_pct


def last_recorded_balance() -> dict[str, float]:
    """Return the latest cached balance information."""

    with LOCK:
        return dict(_balance_snapshot)


def in_cooldown(symbol: str) -> bool:
    """Return ``True`` if ``symbol`` was closed recently and is cooling
    down."""
    norm = normalize_symbol(symbol)
    ts = _last_closed.get(norm)
    if ts is None:
        _trace("cooldown check: %s not found", norm)
        return False
    in_window = (time.time() - ts) < config.TRADE_COOLDOWN
    _trace("cooldown check: %s in_window=%s", norm, in_window)
    return in_window


def _timestamp_to_iso(value: Any) -> str | None:
    """Return ``value`` normalised as an ISO-8601 string or ``None``."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
        return dt.isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    return None


def _resolve_timestamp(*candidates: Any) -> str | None:
    """Return the first valid timestamp amongst ``candidates``."""

    for candidate in candidates:
        ts = _timestamp_to_iso(candidate)
        if ts:
            return ts
    return None


# --- Core functions ---


def add_trade(trade, *, allow_duplicates: bool = False):
    """Añade una nueva operación a la lista de abiertas.

    ``allow_duplicates`` permite explícitamente registrar múltiples operaciones
    para el mismo símbolo. Por defecto se rechazan duplicados para evitar
    inconsistencias si la estrategia genera señales repetidas.
    """

    with LOCK:
        raw_symbol = trade.get("symbol", "")
        trade["symbol"] = normalize_symbol(raw_symbol)
        trade.setdefault("raw_symbol", raw_symbol)
        duplicate = False
        for existing in open_trades:
            if normalize_symbol(existing.get("symbol", "")) == trade["symbol"]:
                duplicate = True
                break
        if duplicate and not allow_duplicates:
            _trace("add_trade rejected duplicate symbol=%s", trade["symbol"])
            raise ValueError(f"Ya existe una operación abierta para {trade['symbol']}")
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
        try:
            entry_price = float(trade.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        trade.setdefault("peak_price", entry_price)
        trade.setdefault("trough_price", entry_price if entry_price > 0 else 0.0)
        open_trades.append(trade)
        log_history("open", trade)
        if config.TECH_ANALYSIS_ENABLED and config.TECH_ANALYSIS_ON_TRADE_EVENTS:
            try:
                from trading_bot.analysis_worker import ANALYSIS_WORKER

                ANALYSIS_WORKER.enqueue(trade.get("symbol", ""), reason="trade_open")
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("No se pudo encolar análisis técnico: %s", exc)
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


def _position_size(position: dict) -> float:
    """Return the absolute size from a position payload."""

    for key in ("holdVolume", "contracts", "positionAmt"):
        try:
            qty = float(position.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if qty != 0:
            return abs(qty)
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
        closed_at = datetime.now(timezone.utc)
        trade["close_time"] = _resolve_timestamp(
            trade.get("close_time"),
            trade.get("close_time_dt"),
            closed_at,
        )
        open_time = _resolve_timestamp(
            trade.get("open_time"),
            trade.get("open_time_dt"),
            trade.get("opened_at"),
            trade.get("created_at"),
            trade.get("created_ts"),
        )
        if open_time:
            trade["open_time"] = open_time
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
        _prune_closed_trades()
        log_history("close", trade)
        try:
            from trading_bot import history

            history.append_trade(trade)
        except Exception as exc:  # pragma: no cover - defensive persistence
            logger.error("Error al guardar historial CSV: %s", exc)
        if config.CLEAR_CLOSED_TRADES_AFTER_EXPORT:
            closed_trades[:] = [
                t for t in closed_trades if t.get("trade_id") != trade.get("trade_id")
            ]
        profit_value = _ensure_realized_pnl(trade)
        try:
            rl_agent.record_trade_outcome(trade)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("No se pudo registrar el trade para RL: %s", exc)
        _notify_profit_observers(profit_value, trade)
        if config.TECH_ANALYSIS_ENABLED and config.TECH_ANALYSIS_ON_TRADE_EVENTS:
            try:
                from trading_bot.analysis_worker import ANALYSIS_WORKER

                ANALYSIS_WORKER.enqueue(trade.get("symbol", ""), reason="trade_close")
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("No se pudo encolar análisis técnico: %s", exc)
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
            _prune_closed_trades()
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
                _prune_closed_trades()
    except Exception as e:
        logger.error("Error cargando trades: %s", e)


# --- Position reconciliation ---


def get_live_position(symbol: str) -> dict | None:
    """Return the current live position for ``symbol`` if present."""

    norm = normalize_symbol(symbol)
    try:
        from . import execution

        positions = execution.fetch_positions()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to fetch live positions for %s: %s", symbol, exc)
        return None

    for pos in positions:
        if normalize_symbol(pos.get("symbol", "")) == norm:
            return pos
    return None


def _position_to_trade(position: dict) -> dict | None:
    symbol = normalize_symbol(position.get("symbol", ""))
    if not symbol:
        return None

    qty = _position_size(position)
    if qty == 0:
        return None

    side_raw = str(position.get("side", "")).lower()
    is_long = side_raw == "long"
    try:
        entry_price = float(position.get("entryPrice") or 0.0)
    except (TypeError, ValueError):
        entry_price = 0.0
    try:
        stop_loss_exchange = float(position.get("stopLossPrice") or 0.0)
    except (TypeError, ValueError):
        stop_loss_exchange = 0.0
    try:
        take_profit_exchange = float(position.get("takeProfitPrice") or 0.0)
    except (TypeError, ValueError):
        take_profit_exchange = 0.0

    stop_loss_fallback = entry_price * (0.98 if is_long else 1.02) if entry_price > 0 else 0.0
    take_profit_fallback = entry_price * (1.02 if is_long else 0.98) if entry_price > 0 else 0.0

    trade = {
        "symbol": symbol,
        "side": "BUY" if is_long else "SELL",
        "quantity": qty,
        "quantity_remaining": qty,
        "requested_quantity": qty,
        "original_quantity": qty,
        "entry_price": entry_price,
        "stop_loss": stop_loss_exchange or stop_loss_fallback,
        "take_profit": take_profit_exchange or take_profit_fallback,
        "leverage": int(position.get("leverage", config.DEFAULT_LEVERAGE)),
        "status": "active",
        "state": TradeState.OPEN.value,
        "open_time": _resolve_timestamp(position.get("timestamp"), position.get("datetime")),
        "closing": False,
    }
    order_info = position.get("info", {})
    if isinstance(order_info, dict) and order_info.get("orderId"):
        trade["order_id"] = order_info.get("orderId")
    return trade


def _extract_unrealized_pnl(position: dict) -> float | None:
    candidates: list[Any] = [position.get("unrealizedPnl"), position.get("upl")]
    info = position.get("info", {})
    if isinstance(info, dict):
        candidates.append(info.get("unrealizedPnl"))
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        return value
    return None


def _extract_realized_pnl(position: dict) -> float | None:
    candidates: list[Any] = [
        position.get("realizedPnl"),
        position.get("closeProfit"),
        position.get("pnl"),
    ]
    info = position.get("info", {})
    if isinstance(info, dict):
        candidates.extend([info.get("realizedPnl"), info.get("closeProfit")])
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        return value
    return None


def _apply_reported_realized_pnl(trade: dict, reported: float | None) -> None:
    if reported is None:
        return

    try:
        current = float(trade.get("realized_pnl") or trade.get("profit") or 0.0)
    except (TypeError, ValueError):
        current = 0.0
    tolerance = max(0.01, abs(current) * 0.05, abs(reported) * 0.05)
    if abs(reported - current) > tolerance:
        logger.warning(
            "[SYNC] Realized PnL mismatch for %s: local=%.6f exchange=%.6f",
            trade.get("symbol"),
            current,
            reported,
        )
    with LOCK:
        trade["realized_pnl"] = reported
        trade["profit"] = reported


def _verify_live_position(trade: dict, expected_qty: float | None = None) -> float | None:
    if config.DRY_RUN:
        return None

    from . import execution

    symbol = trade.get("symbol")
    if not symbol:
        return None
    try:
        expected = float(expected_qty if expected_qty is not None else trade.get("quantity") or 0.0)
    except (TypeError, ValueError):
        expected = 0.0

    live_size = execution.fetch_position_size(symbol)
    mismatch = expected > 0 and abs(live_size - expected) > max(1e-6, expected * 0.01)
    with LOCK:
        trade["last_synced_size"] = live_size
        trade["position_mismatch"] = mismatch
    if mismatch:
        logger.warning(
            "[SYNC] Tamaño en exchange %.8f difiere de esperado %.8f para %s",
            live_size,
            expected,
            symbol,
        )
    else:
        logger.debug("[SYNC] Tamaño confirmado para %s: %.8f", symbol, live_size)
    return live_size


def reconcile_positions() -> None:
    """Synchronise open trades with live positions on the exchange."""

    if config.DRY_RUN:
        return

    from . import execution

    positions = execution.fetch_positions()
    open_orders = execution.fetch_open_orders()
    positions_by_symbol: dict[str, dict] = {}
    for pos in positions:
        symbol = normalize_symbol(pos.get("symbol", ""))
        qty = _position_size(pos)
        if not symbol or qty == 0:
            continue
        positions_by_symbol[symbol] = pos

    for order in open_orders:
        order_id = str(order.get("id") or order.get("orderId") or "")
        symbol = normalize_symbol(order.get("symbol", ""))
        if symbol and find_trade(symbol=symbol) is None:
            logger.warning(
                "[RECONCILE] Open order on exchange without local trade: %s (%s)",
                symbol,
                order_id,
            )

    active_symbols = set(positions_by_symbol.keys())

    for symbol, pos in positions_by_symbol.items():
        trade = find_trade(symbol=symbol)
        qty = _position_size(pos)
        try:
            entry_price = float(pos.get("entryPrice") or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0
        leverage_raw = pos.get("leverage", config.DEFAULT_LEVERAGE)
        try:
            leverage = int(leverage_raw)
        except (TypeError, ValueError):
            leverage = config.DEFAULT_LEVERAGE

        if trade is None:
            recovered = _position_to_trade(pos)
            if not recovered:
                continue
            try:
                add_trade(recovered)
            except ValueError:
                logger.debug("Operacion ya registrada para %s, omitiendo duplicado", symbol)
            else:
                logger.info("[RECONCILE] Recovered external position: %s", symbol)
            continue

        if trade.get("state") == TradeState.PENDING.value:
            logger.info(
                "[RECONCILE] Pending trade for %s confirmed by live position; marking OPEN",
                symbol,
            )
            set_trade_state(trade.get("trade_id"), TradeState.OPEN)

        updates = {
            "quantity": qty,
            "quantity_remaining": qty,
            "entry_price": entry_price if entry_price > 0 else trade.get("entry_price"),
            "leverage": leverage,
            "status": "active",
        }
        unrealized = _extract_unrealized_pnl(pos)
        if unrealized is not None:
            updates["unrealized_pnl"] = unrealized
        order_info = pos.get("info", {})
        if isinstance(order_info, dict) and order_info.get("orderId"):
            updates["order_id"] = order_info.get("orderId")
        update_trade(trade.get("trade_id"), **updates)
        set_trade_state(trade.get("trade_id"), TradeState.OPEN)
        logger.info("[RECONCILE] Synced position for %s: size=%.8f", symbol, qty)

    for tr in list(all_open_trades()):
        symbol = normalize_symbol(tr.get("symbol", ""))
        if symbol in active_symbols:
            continue

        order_id = str(tr.get("order_id") or "")
        created_ts = float(tr.get("created_ts") or 0)
        age_seconds = time.time() - created_ts
        grace_seconds = max(0, int(getattr(config, "ORDER_MAX_AGE", 0)))

        try:
            state = TradeState(tr.get("state"))
        except ValueError:
            state = TradeState.PENDING

        missing_checks = int(tr.get("missing_position_checks") or 0) + 1
        update_trade(tr.get("trade_id"), missing_position_checks=missing_checks)

        status = ""
        confirmed_cancel = False
        if order_id and not config.DRY_RUN:
            try:
                from . import execution  # Import local para evitar ciclos

                order_status = execution.fetch_order_status(order_id, symbol)
                status = str(order_status.get("status") or order_status.get("state") or "").lower()
                confirmed_cancel = status in {"canceled", "cancelled", "rejected", "expired"}
                if confirmed_cancel:
                    logger.warning(
                        "Orden %s reportada como %s sin posición activa; esperando confirmación adicional.",
                        order_id,
                        status,
                    )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Error verificando orden %s: %s. Conservamos trade y marcamos para revisión.",
                    order_id,
                    e,
                )

        review_reason = None
        if confirmed_cancel and missing_checks >= 2:
            if state == TradeState.PENDING:
                cancel_pending_trade(
                    tr.get("trade_id"),
                    reason=f"{status or 'cancelled'}_confirmed_no_position",
                )
                continue
            if state in {TradeState.OPEN, TradeState.PARTIALLY_FILLED, TradeState.CLOSING}:
                review_reason = (
                    f"order_{status}_without_position" if status else "order_cancelled_without_position"
                )
        else:
            review_reason = "missing_position_unconfirmed"

        if not order_id and age_seconds >= grace_seconds:
            review_reason = "missing_order_reference_after_grace"

        if review_reason:
            already_flagged = bool(tr.get("manual_review"))
            update_trade(
                tr.get("trade_id"),
                manual_review=True,
                manual_review_reason=review_reason,
            )
            if not already_flagged:
                logger.error(
                    "[RECONCILE] Trade %s sin posición confirmada (estado=%s, orden=%s). Requiere revisión manual.",
                    symbol,
                    state.value,
                    status or "desconocido",
                )


def start_periodic_position_reconciliation(
    interval_seconds: int = POSITION_RECONCILE_INTERVAL,
) -> threading.Thread:
    """Launch a daemon thread that reconciles positions every ``interval_seconds``."""

    global _reconcile_thread
    if _reconcile_thread is not None and _reconcile_thread.is_alive():
        return _reconcile_thread

    def _loop():
        while True:
            try:
                reconcile_positions()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Periodic reconcile error: %s", exc)
            time.sleep(max(1, interval_seconds))

    thread = threading.Thread(target=_loop, daemon=True, name="trade-reconcile")
    thread.start()
    _reconcile_thread = thread
    return thread


def start_balance_monitoring(
    interval_seconds: int = BALANCE_MONITOR_INTERVAL,
) -> threading.Thread | None:
    """Launch a daemon thread that refreshes cached balance."""

    if config.DRY_RUN:
        return None

    global _balance_thread
    if _balance_thread is not None and _balance_thread.is_alive():
        return _balance_thread

    def _loop():
        from . import execution

        while True:
            try:
                bal = execution.fetch_balance()
                _record_balance_snapshot(bal)
                logger.debug("[BALANCE] Saldo actualizado: %.4f USDT", bal)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Error actualizando balance: %s", exc)
            time.sleep(max(5, interval_seconds))

    thread = threading.Thread(target=_loop, daemon=True, name="balance-monitor")
    thread.start()
    _balance_thread = thread
    return thread


# --- WebSocket helpers and handlers ---


def _ws_order_to_trade(order: dict) -> dict | None:
    symbol = normalize_symbol(order.get("symbol", ""))
    if not symbol:
        return None

    for qty_key in ("size", "filledSize", "filledQty"):
        try:
            qty = abs(float(order.get(qty_key) or 0.0))
        except (TypeError, ValueError):
            continue
        if qty != 0:
            break
    else:
        qty = 0.0

    try:
        entry_price = float(order.get("price") or order.get("avgPrice") or order.get("fillPrice") or 0.0)
    except (TypeError, ValueError):
        entry_price = 0.0

    if qty == 0:
        return None

    side_raw = str(order.get("side", "")).lower()
    is_buy = side_raw in {"buy", "long"}
    trade = {
        "symbol": symbol,
        "side": "BUY" if is_buy else "SELL",
        "quantity": qty,
        "quantity_remaining": qty,
        "requested_quantity": qty,
        "original_quantity": qty,
        "entry_price": entry_price,
        "leverage": config.DEFAULT_LEVERAGE,
        "status": "active",
        "state": TradeState.OPEN.value,
        "open_time": _resolve_timestamp(order.get("timestamp"), order.get("cTime")),
        "closing": False,
    }
    order_id = order.get("orderId") or order.get("id")
    if order_id:
        trade["order_id"] = order_id
    return trade


def create_trade_from_position(position: dict) -> dict | None:
    trade = _position_to_trade(position)
    if not trade:
        return None
    try:
        return add_trade(trade)
    except ValueError:
        return find_trade(symbol=trade.get("symbol"))


def create_trade_from_ws(order: dict) -> dict | None:
    trade = _ws_order_to_trade(order)
    if not trade:
        return None
    try:
        return add_trade(trade)
    except ValueError:
        return find_trade(symbol=trade.get("symbol"))


def verify_trade_on_exchange(trade_id: str, expected_qty: float | None = None) -> float | None:
    """Cross-check a trade size with the exchange position size."""

    trade = get_open_trade(trade_id)
    if not trade:
        return None
    return _verify_live_position(trade, expected_qty)


def _current_state(trade: dict) -> TradeState:
    try:
        return TradeState(trade.get("state"))
    except ValueError:
        return TradeState.PENDING


def _update_from_ws(trade: dict, *, quantity: float | None, entry_price: float | None, state: TradeState) -> None:
    updates = {}
    if quantity is not None and quantity > 0:
        updates["quantity"] = quantity
        requested = _original_quantity(trade)
        remaining = max(requested - quantity, 0.0) if requested > 0 else quantity
        updates["quantity_remaining"] = remaining
        updates["status"] = "active"
    if entry_price is not None and entry_price > 0:
        updates["entry_price"] = entry_price
    if updates:
        update_trade(trade.get("trade_id"), **updates)
    set_trade_state(trade.get("trade_id"), state)


def ws_order_filled(order):
    symbol = normalize_symbol(order.get("symbol", ""))
    if not symbol:
        return

    trade = find_trade(symbol=symbol)
    if trade is None:
        trade = create_trade_from_ws(order)
    if trade is None:
        logger.debug("[WS] Ignoring filled order without trade for %s", symbol)
        return

    state = _current_state(trade)

    try:
        qty = abs(float(order.get("size") or order.get("filledSize") or order.get("filledQty") or 0.0))
    except (TypeError, ValueError):
        qty = None
    try:
        price = float(order.get("price") or order.get("avgPrice") or order.get("fillPrice") or 0.0)
    except (TypeError, ValueError):
        price = None

    if state == TradeState.OPEN:
        try:
            existing_qty = float(trade.get("quantity") or 0.0)
        except (TypeError, ValueError):
            existing_qty = 0.0

        if qty is None or abs(existing_qty - qty) < 1e-8:
            logger.debug("[WS] Duplicate fill for %s ignored", symbol)
            return

    _update_from_ws(trade, quantity=qty, entry_price=price, state=TradeState.OPEN)
    logger.info("[WS] Trade OPEN via WS %s", symbol)


def ws_order_partial(order):
    symbol = normalize_symbol(order.get("symbol", ""))
    if not symbol:
        return

    trade = find_trade(symbol=symbol)
    if trade is None:
        logger.debug("[WS] Partial fill without existing trade for %s", symbol)
        return
    state = _current_state(trade)
    try:
        qty = abs(float(order.get("filledSize") or order.get("filledQty") or order.get("size") or 0.0))
    except (TypeError, ValueError):
        qty = None
    if state in {TradeState.PARTIALLY_FILLED, TradeState.OPEN}:
        try:
            existing_qty = float(trade.get("quantity") or 0.0)
        except (TypeError, ValueError):
            existing_qty = 0.0

        if qty is None or qty <= existing_qty:
            logger.debug("[WS] Duplicate partial fill for %s ignored", symbol)
            return
    _update_from_ws(trade, quantity=qty, entry_price=None, state=TradeState.PARTIALLY_FILLED)
    logger.info("[WS] Trade partial fill %s", symbol)


def ws_order_cancelled(order):
    symbol = normalize_symbol(order.get("symbol", ""))
    if not symbol:
        return
    trade = find_trade(symbol=symbol)
    if trade:
        state = _current_state(trade)
        if state in {TradeState.CANCELLED, TradeState.FAILED, TradeState.CLOSED}:
            logger.debug("[WS] Duplicate cancel for %s ignored", symbol)
            return
        cancel_pending_trade(trade.get("trade_id"), reason="cancelled_by_exchange")
        logger.warning("[WS] Trade CANCELLED %s", symbol)


def ws_position_update(pos):
    symbol = normalize_symbol(pos.get("symbol", ""))
    if not symbol:
        return

    size = _position_size(pos)
    if size == 0:
        return ws_position_closed(pos)

    trade = find_trade(symbol=symbol)
    if trade is None:
        trade = create_trade_from_position(pos)
    if trade is None:
        logger.debug("[WS] Ignoring position update without trade for %s", symbol)
        return

    updates = {
        "quantity": size,
        "quantity_remaining": size,
        "status": "active",
    }
    try:
        entry_price = float(pos.get("entryPrice") or 0.0)
    except (TypeError, ValueError):
        entry_price = 0.0
    if entry_price > 0:
        updates["entry_price"] = entry_price
    unrealized = _extract_unrealized_pnl(pos)
    if unrealized is not None:
        updates["unrealized_pnl"] = unrealized

    update_trade(trade.get("trade_id"), **updates)
    set_trade_state(trade.get("trade_id"), TradeState.OPEN)
    logger.info("[WS] Sync pos %s", symbol)


def ws_position_closed(pos):
    symbol = normalize_symbol(pos.get("symbol", ""))
    if not symbol:
        return

    trade = find_trade(symbol=symbol)
    if trade:
        state = _current_state(trade)
        if state in {TradeState.CLOSED, TradeState.FAILED}:
            logger.debug("[WS] Duplicate position closed for %s ignored", symbol)
            return
        reported_pnl = _extract_realized_pnl(pos)
        try:
            exit_price = float(data.get_current_price_ticker(symbol) or 0.0)
        except (TypeError, ValueError):
            exit_price = None
        closed = close_trade(
            trade_id=trade.get("trade_id"),
            reason="ws_position_closed",
            exit_price=exit_price,
            profit=reported_pnl,
        )
        if closed is not None:
            _apply_reported_realized_pnl(closed, reported_pnl)
        logger.info("[WS] Position CLOSED %s", symbol)

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
