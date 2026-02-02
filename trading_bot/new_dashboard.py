"""Minimal futuristic dashboard blueprint and server."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
from pathlib import Path
from typing import Any

from flask import Blueprint, Flask, abort, jsonify, render_template, request, send_from_directory

from . import config, data, history, trade_manager
from .analysis_store import GLOBAL_ANALYSIS_STORE
from .analysis_worker import ANALYSIS_WORKER


dashboard_bp = Blueprint("dashboard", __name__, template_folder="templates")


@dashboard_bp.route("/")
def landing():
    """Serve the trading dashboard page."""

    return render_template("dashboard_hero.html")


@dashboard_bp.route("/index.html")
@dashboard_bp.route("/dashboard")
def landing_alias():
    """Provide common landing aliases for the dashboard."""

    return render_template("dashboard_hero.html")


@dashboard_bp.route("/<path:subpath>")
def catch_all(subpath: str):
    """Fallback to the dashboard for client-side routes."""

    if subpath.startswith("api/"):
        abort(404)
    return render_template("dashboard_hero.html")


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    side: str | None
    quantity: float | None
    entry_price: float | None
    last_price: float | None
    take_profit: float | None
    stop_loss: float | None
    unrealized_pnl: float | None
    status: str | None
    open_time: str | None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_iso(timestamp: Any) -> str | None:
    if timestamp is None or timestamp == "":
        return None
    if isinstance(timestamp, str):
        return timestamp
    try:
        dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _snapshot_position(trade: dict, last_price: float | None) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=trade.get("symbol", ""),
        side=trade.get("side"),
        quantity=_to_float(trade.get("quantity") or trade.get("quantity_remaining")),
        entry_price=_to_float(trade.get("entry_price")),
        last_price=last_price,
        take_profit=_to_float(trade.get("take_profit")),
        stop_loss=_to_float(trade.get("stop_loss")),
        unrealized_pnl=_to_float(trade.get("unrealized_pnl") or trade.get("profit")),
        status=trade.get("status"),
        open_time=_to_iso(trade.get("open_time")),
    )


def _read_trade_history(limit: int = 200) -> list[dict[str, Any]]:
    path: Path = history.HISTORY_FILE
    if not path.exists():
        return []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if limit > 0:
        rows = rows[-limit:]
    return rows


def create_app() -> Flask:
    """Build a lightweight Flask app that only serves the new dashboard."""

    app = Flask(__name__)
    app.register_blueprint(dashboard_bp)

    @app.get("/api/health")
    def api_health():
        return jsonify({"ok": True})

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("dashboard_hero.html"), 200

    @app.get("/api/status")
    def api_status():
        snapshot = trade_manager.last_recorded_balance()
        return jsonify(
            {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "bot_mode": config.BOT_MODE,
                "trading_mode": config.TRADING_MODE,
                "trading_enabled": config.ENABLE_TRADING,
                "open_positions": trade_manager.count_open_trades(),
                "balance": snapshot,
            }
        )

    @app.get("/api/positions")
    def api_positions():
        include_prices = request.args.get("include_prices", "1") != "0"
        with trade_manager.LOCK:
            trades = list(trade_manager.open_trades)
        snapshots: list[PositionSnapshot] = []
        for trade in trades:
            price = data.get_current_price_ticker(trade.get("symbol", "")) if include_prices else None
            snapshots.append(_snapshot_position(trade, price))
        return jsonify([asdict(snapshot) for snapshot in snapshots])

    @app.get("/api/price")
    def api_price():
        symbol = request.args.get("symbol", "")
        price = data.get_current_price_ticker(symbol) if symbol else None
        return jsonify({"symbol": symbol, "price": price})

    @app.get("/api/history")
    def api_history():
        limit = request.args.get("limit", "200")
        try:
            limit_value = max(0, int(limit))
        except ValueError:
            limit_value = 200
        return jsonify(_read_trade_history(limit_value))

    @app.get("/api/decisions")
    def api_decisions():
        events = trade_manager.get_history()
        return jsonify(events[-200:])

    @app.get("/api/technical_analysis")
    def api_technical_analysis():
        return jsonify(GLOBAL_ANALYSIS_STORE.list(limit=50))

    @app.get("/api/technical_analysis/<symbol>")
    def api_technical_analysis_latest(symbol: str):
        report = GLOBAL_ANALYSIS_STORE.latest_for(symbol)
        return jsonify(report or {})

    @app.post("/api/technical_analysis/<symbol>/run")
    def api_technical_analysis_run(symbol: str):
        ok = ANALYSIS_WORKER.enqueue(symbol, reason="manual")
        return jsonify({"queued": ok})

    @app.get("/charts/<path:filename>")
    def charts(filename: str):
        return send_from_directory("charts", filename)

    return app


def start_dashboard(host: str, port: int) -> None:
    """Launch the minimal dashboard server."""

    app = create_app()
    app.run(host=host, port=port, use_reloader=False)
