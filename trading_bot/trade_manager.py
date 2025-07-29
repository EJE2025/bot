# trade_manager.py

import threading
import json
import os
from datetime import datetime
import time
import logging
from . import config
from .utils import normalize_symbol

logger = logging.getLogger(__name__)

open_trades = []
closed_trades = []
trade_history = []  # Guarda todos los cambios si quieres auditar

LOCK = threading.Lock()

# Cool-down registry for recently closed symbols
_last_closed: dict[str, float] = {}


def normalize_symbol(symbol: str) -> str:
    """Return a normalized symbol like ``BTC_USDT`` regardless of separators."""
    raw = symbol.replace("/", "").replace(":USDT", "").replace("_", "")
    raw = raw.upper()
    if not raw.endswith("USDT"):
        raw += "USDT"
    base = raw[:-4]
    return f"{base}_USDT"

# --- Core functions ---

def add_trade(trade):
    """Añade una nueva operación a la lista de abiertas."""
    with LOCK:
        trade["symbol"] = normalize_symbol(trade.get("symbol", ""))
        if "trade_id" not in trade:
            trade["trade_id"] = f"{trade['symbol']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        trade.setdefault("open_time", datetime.utcnow().isoformat())
        trade.setdefault("status", "pending")
        open_trades.append(trade)
        log_history("open", trade)


def find_trade(symbol=None, trade_id=None):
    """Devuelve la primera operación abierta que coincida con el símbolo o el ID."""
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


def close_trade(trade_id=None, symbol=None, reason="closed", exit_price=None, profit=None):
    """Cierra una operación y la mueve a cerradas, añadiendo motivo.

    Parámetros adicionales permiten registrar precio de salida y beneficio
    para que el historial en JSON tenga la misma información que el CSV de
    `history`.
    """
    norm = normalize_symbol(symbol) if symbol else None
    with LOCK:
        idx = None
        for i, trade in enumerate(open_trades):
            if (trade_id and trade.get("trade_id") == trade_id) or (
                norm and normalize_symbol(trade.get("symbol")) == norm
            ):
                idx = i
                break
        if idx is not None:
            trade = open_trades.pop(idx)
            _last_closed[normalize_symbol(trade.get("symbol"))] = time.time()
            trade["close_time"] = datetime.utcnow().isoformat()
            trade["close_reason"] = reason
            if exit_price is not None:
                trade["exit_price"] = exit_price
            if profit is not None:
                trade["profit"] = profit
            trade["status"] = "closed"
            closed_trades.append(trade)
            log_history("close", trade)
            return trade
    return None


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


def save_trades(open_path="open_trades.json", closed_path="closed_trades.json"):
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


def load_trades(open_path="open_trades.json", closed_path="closed_trades.json"):
    try:
        if os.path.exists(open_path):
            with open(open_path, "r") as f:
                data = json.load(f)
            with LOCK:
                open_trades.clear()
                for t in data:
                    t.setdefault("status", "active")
                open_trades.extend(data)
        if os.path.exists(closed_path):
            with open(closed_path, "r") as f:
                data = json.load(f)
            with LOCK:
                closed_trades.clear()
                for t in data:
                    t.setdefault("status", "closed")
                closed_trades.extend(data)
    except Exception as e:
        logger.error("Error cargando trades: %s", e)

# --- Optional: Auditing/history ---

def log_history(event_type, trade):
    """Store a snapshot of the trade change if history logging is enabled."""
    if not config.ENABLE_TRADE_HISTORY_LOG:
        return
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
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
