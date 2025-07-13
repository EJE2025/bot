# trade_manager.py

import threading
import json
import os
from datetime import datetime

open_trades = []
closed_trades = []
trade_history = []  # Guarda todos los cambios si quieres auditar

LOCK = threading.Lock()

# --- Core functions ---

def add_trade(trade):
    """Añade una nueva operación a la lista de abiertas."""
    with LOCK:
        if "trade_id" not in trade:
            trade["trade_id"] = f"{trade['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        trade.setdefault("open_time", datetime.now().isoformat())
        trade.setdefault("status", "pending")
        open_trades.append(trade)
        log_history("open", trade)


def find_trade(symbol=None, trade_id=None):
    """Busca una operación abierta por símbolo o ID."""
    with LOCK:
        for trade in open_trades:
            if (symbol and trade.get("symbol") == symbol) or (trade_id and trade.get("trade_id") == trade_id):
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


def close_trade(trade_id=None, symbol=None, reason="closed"):
    """Cierra una operación y la mueve a cerradas, añadiendo motivo."""
    with LOCK:
        idx = None
        for i, trade in enumerate(open_trades):
            if (trade_id and trade.get("trade_id") == trade_id) or (symbol and trade.get("symbol") == symbol):
                idx = i
                break
        if idx is not None:
            trade = open_trades.pop(idx)
            trade["close_time"] = datetime.now().isoformat()
            trade["close_reason"] = reason
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

def _atomic_write(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_trades(open_path="open_trades.json", closed_path="closed_trades.json"):
    try:
        with LOCK:
            _atomic_write(open_path, open_trades)
            _atomic_write(closed_path, closed_trades)
    except Exception as e:
        print(f"[trade_manager] Error guardando trades: {e}")


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
        print(f"[trade_manager] Error cargando trades: {e}")

# --- Optional: Auditing/history ---

def log_history(event_type, trade):
    # Guarda una copia del cambio (puedes desactivar si no quieres logs grandes)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "trade_snapshot": trade.copy()
    }
    trade_history.append(entry)


def get_history():
    return list(trade_history)

# --- Utilities ---

def get_trades_by_symbol(symbol):
    with LOCK:
        return [t for t in open_trades if t.get("symbol") == symbol]


def count_open_trades():
    with LOCK:
        return len(open_trades)
