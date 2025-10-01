from __future__ import annotations

import csv
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

try:
    from flask import Flask, render_template, jsonify, request
except ImportError:  # Flask not installed
    Flask = None

try:
    from flask_socketio import SocketIO
except ImportError:  # SocketIO optional
    SocketIO = None

from trading_bot.trade_manager import (
    all_open_trades,
    close_trade_full,
    close_trade_partial,
    get_open_trade,
)
from trading_bot import liquidity_ws, data
from trading_bot.history import HISTORY_FILE, FIELDS as HISTORY_FIELDS

if Flask:
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*") if SocketIO else None

    _trade_locks: dict[str, threading.Lock] = {}
    _locks_lock = threading.Lock()

    def _get_trade_lock(trade_id: str) -> threading.Lock:
        with _locks_lock:
            if trade_id not in _trade_locks:
                _trade_locks[trade_id] = threading.Lock()
            return _trade_locks[trade_id]

    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _trades_with_metrics() -> list[dict[str, Any]]:
        trades: list[dict[str, Any]] = []
        for trade in all_open_trades():
            sym = trade.get("symbol")
            entry = _coerce_float(trade.get("entry_price", 0))
            qty = _coerce_float(trade.get("quantity_remaining", trade.get("quantity", 0)))
            side = str(trade.get("side", "BUY")).upper()
            current_price = data.get_current_price_ticker(sym)
            if not current_price:
                current_price = entry
            pnl = (
                (current_price - entry) * qty
                if side == "BUY"
                else (entry - current_price) * qty
            )
            row = dict(trade)
            row["entry_price"] = entry
            row["quantity"] = qty
            row["quantity_remaining"] = qty
            row["current_price"] = current_price
            row["pnl_unrealized"] = pnl
            row["notional_value"] = abs(entry * qty)
            row["realized_pnl"] = _coerce_float(trade.get("realized_pnl"))
            trades.append(row)
        return trades

    @app.route("/")
    def index():
        return render_template("index.html", current_year=datetime.utcnow().year)

    @app.route("/api/trades")
    def api_trades():
        """Return open trades with current price and unrealized PnL."""
        return jsonify(_trades_with_metrics())

    def _emit(event: str, payload: Any) -> None:
        if socketio:
            socketio.emit(event, payload, namespace="/ws")

    @app.post("/api/trades/<trade_id>/close")
    def api_close_trade(trade_id: str):
        lock = _get_trade_lock(trade_id)
        with lock:
            trade = get_open_trade(trade_id)
            if not trade:
                closed = close_trade_full(trade_id)
                if not closed:
                    return jsonify({"ok": False, "error": "Trade no encontrado"}), 404
                return jsonify({"ok": True, "trade": closed, "note": "ya estaba cerrado"})
            payload = request.get_json(silent=True) or {}
            reason = payload.get("reason", "manual_close")
            result = close_trade_full(trade_id, reason=reason)
            if not result:
                return jsonify({"ok": False, "error": "No se pudo cerrar la operación"}), 400
        _emit("trade_closed", {"trade": result})
        _emit("trades_refresh", _trades_with_metrics())
        return jsonify({"ok": True, "trade": result})

    @app.post("/api/trades/<trade_id>/close-partial")
    def api_close_trade_partial(trade_id: str):
        lock = _get_trade_lock(trade_id)
        with lock:
            trade = get_open_trade(trade_id)
            if not trade:
                return jsonify({"ok": False, "error": "Trade no encontrado o ya cerrado"}), 404

            data_payload = request.get_json(silent=True) or {}
            qty = data_payload.get("qty")
            pct = data_payload.get("percent")
            reason = data_payload.get("reason", "manual_partial")

            if qty is None and pct is None:
                return jsonify({"ok": False, "error": "Debes informar qty o percent"}), 400

            remaining = _coerce_float(trade.get("quantity_remaining", trade.get("quantity", 0)))
            if pct is not None:
                try:
                    pct_val = float(pct)
                except (TypeError, ValueError):
                    return jsonify({"ok": False, "error": "percent inválido"}), 400
                if pct_val <= 0 or pct_val > 100:
                    return jsonify({"ok": False, "error": "percent fuera de rango (1-100)"}), 400
                qty = remaining * (pct_val / 100.0)

            try:
                qty_val = float(qty)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "qty inválido"}), 400
            if qty_val <= 0:
                return jsonify({"ok": False, "error": "qty debe ser > 0"}), 400
            if qty_val > remaining + 1e-12:
                return jsonify({"ok": False, "error": "qty excede la cantidad restante"}), 400

            try:
                result = close_trade_partial(trade_id, quantity=qty_val, reason=reason)
            except ValueError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            if not result:
                return jsonify({"ok": False, "error": "No se pudo cerrar parcialmente"}), 400
        _emit("trade_updated", {"trade": result})
        _emit("trades_refresh", _trades_with_metrics())
        return jsonify({"ok": True, "trade": result})

    @app.route("/api/summary")
    def api_summary():
        """Aggregate metrics that power the dashboard widgets."""
        trades = _trades_with_metrics()
        total_positions = len(trades)
        total_exposure = sum(abs(t["quantity_remaining"]) for t in trades)
        gross_notional = sum(t["notional_value"] for t in trades)
        unrealized_pnl = sum(t["pnl_unrealized"] for t in trades)
        winners = sum(1 for t in trades if t["pnl_unrealized"] > 0)
        losers = sum(1 for t in trades if t["pnl_unrealized"] < 0)

        per_symbol: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "symbol": "",
                "positions": 0,
                "exposure": 0.0,
                "unrealized_pnl": 0.0,
                "notional_value": 0.0,
            }
        )

        for trade in trades:
            sym = str(trade.get("symbol", "")).upper()
            entry = per_symbol[sym]
            entry["symbol"] = sym
            entry["positions"] += 1
            entry["exposure"] += abs(trade["quantity_remaining"])
            entry["unrealized_pnl"] += trade["pnl_unrealized"]
            entry["notional_value"] += trade["notional_value"]

        per_symbol_list = sorted(
            per_symbol.values(),
            key=lambda item: item["notional_value"],
            reverse=True,
        )

        win_rate = 0.0
        deciding = winners + losers
        if deciding:
            win_rate = winners / deciding

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_positions": total_positions,
            "total_exposure": total_exposure,
            "gross_notional": gross_notional,
            "unrealized_pnl": unrealized_pnl,
            "winning_positions": winners,
            "losing_positions": losers,
            "win_rate": win_rate,
            "per_symbol": per_symbol_list,
        }
        return jsonify(payload)

    @app.route("/api/history")
    def api_history():
        """Return the latest closed trades from the CSV history file."""
        try:
            limit = int(request.args.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 200))

        rows: list[dict[str, Any]] = []
        if HISTORY_FILE.exists():
            with HISTORY_FILE.open(newline="") as handle:
                reader = list(csv.DictReader(handle, fieldnames=HISTORY_FIELDS))
            if reader:
                # The first row might be the header if DictReader didn't skip it
                if reader[0] and reader[0][HISTORY_FIELDS[0]] == HISTORY_FIELDS[0]:
                    reader = reader[1:]
                rows = reader[-limit:]

        formatted_rows: list[dict[str, Any]] = []
        for row in reversed(rows):
            formatted_rows.append(
                {
                    key: row.get(key, "")
                    for key in (
                        "symbol",
                        "side",
                        "quantity",
                        "entry_price",
                        "exit_price",
                        "profit",
                        "open_time",
                        "close_time",
                    )
                }
            )

        return jsonify(formatted_rows)

    @app.route("/api/liquidity")
    def api_liquidity():
        """Return current liquidity order book data."""
        raw = liquidity_ws.get_liquidity()
        converted: dict[str, dict[str, list[list[float]]]] = {}
        for sym, book in raw.items():
            bids_dict = book.get("bids", {})
            asks_dict = book.get("asks", {})
            bids = sorted(bids_dict.items(), key=lambda x: x[0], reverse=True)
            asks = sorted(asks_dict.items(), key=lambda x: x[0])
            converted[sym] = {
                "bids": [[float(p), float(q)] for p, q in bids],
                "asks": [[float(p), float(q)] for p, q in asks],
            }
        return jsonify(converted)

    def start_dashboard(host: str, port: int):
        """Run the Flask dashboard in real-time with trade data."""
        if socketio:
            socketio.run(app, host=host, port=port)
        else:
            app.run(host=host, port=port)
else:
    def start_dashboard(host: str, port: int):
        raise ImportError("Flask is required to run the dashboard")
