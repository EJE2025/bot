from __future__ import annotations

import csv
import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
except ImportError:  # Flask not installed
    Flask = None

try:
    from flask_socketio import SocketIO
except ImportError:  # SocketIO optional
    SocketIO = None

from trading_bot import config
from trading_bot.trade_manager import (
    all_closed_trades,
    all_open_trades,
    close_trade_full,
    close_trade_partial,
    get_open_trade,
)
from trading_bot import data
from trading_bot.history import HISTORY_FILE, FIELDS as HISTORY_FIELDS

logger = logging.getLogger(__name__)

if Flask:
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*") if SocketIO else None

    _trade_locks: dict[str, threading.Lock] = {}
    _locks_lock = threading.Lock()
    _session_lock = threading.Lock()
    _session_started_at = datetime.utcnow()
    _session_has_positions = False
    _trading_state_lock = threading.Lock()

    _TRADING_STATE_PATH = Path(getattr(config, "__file__", __file__)).with_name(
        "auto_trade_state.json"
    )

    def _write_trading_state(active: bool) -> None:
        """Persist the AUTO_TRADE flag so it survives restarts."""

        path = _TRADING_STATE_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            payload = {
                "auto_trade": bool(active),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except OSError as exc:  # pragma: no cover - disk issues are unlikely in tests
            logger.warning("No se pudo persistir el estado de AUTO_TRADE: %s", exc)

    def _apply_trading_state(active: bool, *, persist: bool = False) -> bool:
        """Update config flags and optionally persist them to disk."""

        active_flag = bool(active)
        with _trading_state_lock:
            config.AUTO_TRADE = active_flag
            config.MAINTENANCE = not active_flag
            if persist:
                _write_trading_state(active_flag)
        return active_flag

    def _load_trading_state() -> bool:
        """Load AUTO_TRADE from disk if available and keep it consistent."""

        stored = bool(getattr(config, "AUTO_TRADE", True))
        path = _TRADING_STATE_PATH
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("No se pudo leer el estado persistido de AUTO_TRADE: %s", exc)
            else:
                if isinstance(raw, dict) and "auto_trade" in raw:
                    stored = bool(raw["auto_trade"])
        return _apply_trading_state(stored, persist=True)

    _load_trading_state()

    def _get_trade_lock(trade_id: str) -> threading.Lock:
        with _locks_lock:
            if trade_id not in _trade_locks:
                _trade_locks[trade_id] = threading.Lock()
            return _trade_locks[trade_id]

    def _session_identifier(has_positions: bool) -> str:
        global _session_started_at, _session_has_positions
        with _session_lock:
            if not has_positions and _session_has_positions:
                _session_started_at = datetime.utcnow()
            elif _session_started_at is None:
                _session_started_at = datetime.utcnow()
            _session_has_positions = has_positions
            started = _session_started_at or datetime.utcnow()
            return started.replace(microsecond=0).isoformat() + "Z"

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
            qty = abs(
                _coerce_float(
                    trade.get("quantity_remaining", trade.get("quantity", 0))
                )
            )
            side = str(trade.get("side", "BUY")).upper()
            current_price_raw = data.get_current_price_ticker(sym)
            if current_price_raw is None:
                current_price_raw = entry
            current_price = _coerce_float(current_price_raw)
            leverage = 1.0
            try:
                leverage = float(trade.get("leverage") or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            if leverage <= 0:
                leverage = 1.0
            invested_value = 0.0
            if entry > 0 and qty > 0:
                invested_value = abs(entry * qty) / leverage
            if side == "BUY":
                pnl = (current_price - entry) * qty / leverage
            else:
                pnl = (entry - current_price) * qty / leverage
            if invested_value > 0:
                pnl = max(pnl, -invested_value)
            row = dict(trade)
            row["entry_price"] = entry
            row["quantity"] = qty
            row["quantity_remaining"] = qty
            row["current_price"] = current_price
            row["pnl_unrealized"] = pnl
            row["invested_value"] = invested_value
            row["realized_pnl"] = _coerce_float(trade.get("realized_pnl"))
            row["leverage"] = leverage
            trades.append(row)
        return trades

    def _realized_aggregates() -> tuple[float, float, int, int]:
        """Compute realized balance, PnL and win/loss counts from closed trades."""

        realized_balance = 0.0
        realized_pnl_total = 0.0
        winners = 0
        losers = 0

        for trade in all_closed_trades():
            realized = _coerce_float(trade.get("realized_pnl") or trade.get("profit"))
            realized_pnl_total += realized
            if realized > 0:
                realized_balance += realized
                winners += 1
            elif realized < 0:
                losers += 1

        return realized_balance, realized_pnl_total, winners, losers

    @app.route("/")
    def index():
        gateway_base = os.getenv("GATEWAY_BASE_URL", "http://localhost:8080")
        analytics_url = os.getenv("ANALYTICS_GRAPHQL_URL") or f"{gateway_base.rstrip('/')}/graphql"
        ai_base = os.getenv("AI_GATEWAY_URL") or gateway_base.rstrip("/")
        return render_template(
            "index.html",
            current_year=datetime.utcnow().year,
            api_base=gateway_base.rstrip("/"),
            graphql_url=analytics_url,
            ai_chat_url=f"{ai_base}/ai/chat",
            ai_report_url=f"{ai_base}/ai/report",
        )

    @app.route("/manifest.json")
    def manifest():
        return send_from_directory(app.static_folder, "manifest.json")

    @app.route("/service-worker.js")
    def service_worker():
        return send_from_directory(app.static_folder, "service-worker.js")

    @app.route("/api/trades")
    def api_trades():
        """Return open trades with current price and unrealized PnL."""
        return jsonify(_trades_with_metrics())

    @app.post("/api/toggle-trading")
    def api_toggle_trading():
        """Permite pausar o reanudar el trading automático desde el dashboard."""

        current = bool(getattr(config, "AUTO_TRADE", True))
        new_status = _apply_trading_state(not current, persist=True)
        state_label = "habilitado" if new_status else "pausado"
        logger.warning("Trading %s por solicitud vía dashboard", state_label)
        payload = {
            "ok": True,
            "trading_active": new_status,
            "status": state_label,
            "persisted": True,
        }
        _emit("bot_status", {"trading_active": new_status})
        return jsonify(payload)

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
                    return jsonify(
                        {"ok": False, "error": "percent inválido (usa un número real)"}
                    ), 400
                if pct_val <= 0 or pct_val > 100:
                    return jsonify(
                        {
                            "ok": False,
                            "error": "percent fuera de rango permitido (1-100)",
                        }
                    ), 400
                qty = remaining * (pct_val / 100.0)

            try:
                qty_val = float(qty)
            except (TypeError, ValueError):
                return jsonify(
                    {
                        "ok": False,
                        "error": "qty inválido (usa un número mayor que 0)",
                    }
                ), 400
            if qty_val <= 0:
                return jsonify(
                    {"ok": False, "error": "qty debe ser mayor que 0"}
                ), 400
            if qty_val > remaining + 1e-12:
                return jsonify(
                    {
                        "ok": False,
                        "error": "qty excede la cantidad restante"
                        f" ({remaining:.6f})",
                    }
                ), 400

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
        total_quantity = sum(abs(t["quantity_remaining"]) for t in trades)
        total_invested = sum(t["invested_value"] for t in trades)
        total_exposure = sum(
            abs(t["quantity_remaining"] * t.get("current_price", 0.0)) for t in trades
        )
        unrealized_pnl = sum(t["pnl_unrealized"] for t in trades)
        open_winners = sum(1 for t in trades if t["pnl_unrealized"] > 0)
        open_losers = sum(1 for t in trades if t["pnl_unrealized"] < 0)
        (
            realized_balance,
            realized_pnl_total,
            closed_winners,
            closed_losers,
        ) = _realized_aggregates()
        session_id = _session_identifier(total_positions > 0)

        per_symbol: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "symbol": "",
                "positions": 0,
                "quantity": 0.0,
                "exposure": 0.0,
                "unrealized_pnl": 0.0,
                "invested_value": 0.0,
            }
        )

        for trade in trades:
            sym = str(trade.get("symbol", "")).upper()
            entry = per_symbol[sym]
            entry["symbol"] = sym
            entry["positions"] += 1
            entry["quantity"] += abs(trade["quantity_remaining"])
            entry["exposure"] += abs(
                trade["quantity_remaining"] * trade.get("current_price", 0.0)
            )
            entry["unrealized_pnl"] += trade["pnl_unrealized"]
            entry["invested_value"] += trade["invested_value"]

        per_symbol_list = sorted(
            per_symbol.values(),
            key=lambda item: item["exposure"],
            reverse=True,
        )

        win_rate: float | None = None
        total_winners = open_winners + closed_winners
        total_losers = open_losers + closed_losers
        deciding = total_winners + total_losers
        if deciding:
            win_rate = total_winners / deciding

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_positions": total_positions,
            "total_exposure": total_exposure,
            "total_invested": total_invested,
            "total_quantity": total_quantity,
            "unrealized_pnl": unrealized_pnl,
            "realized_balance": realized_balance,
            "realized_pnl": realized_pnl_total,
            "realized_pnl_total": realized_pnl_total,
            "total_pnl": unrealized_pnl + realized_pnl_total,
            "winning_positions": open_winners,
            "losing_positions": open_losers,
            "win_rate": win_rate,
            "win_rate_samples": deciding,
            "win_rate_winners": total_winners,
            "win_rate_losers": total_losers,
            "per_symbol": per_symbol_list,
            "session_id": session_id,
            "trading_active": bool(
                getattr(config, "AUTO_TRADE", True)
                and not getattr(config, "MAINTENANCE", False)
            ),
        }
        trading_mode = str(getattr(config, "TRADING_MODE", "live")).lower()
        payload["trading_mode"] = trading_mode
        payload["paper_trading"] = trading_mode != "live"
        return jsonify(payload)

    @app.route("/api/history")
    def api_history():
        """Return the latest closed trades from the CSV history file."""
        try:
            limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            limit = 50
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

    def start_dashboard(host: str, port: int):
        """Run the Flask dashboard in real-time with trade data."""
        if socketio:
            socketio.run(app, host=host, port=port)
        else:
            app.run(host=host, port=port)
else:
    def start_dashboard(host: str, port: int):
        raise ImportError("Flask is required to run the dashboard")
