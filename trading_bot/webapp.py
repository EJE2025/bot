from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from typing import Any

try:
    from flask import Flask, render_template, jsonify, request
except ImportError:  # Flask not installed
    Flask = None

from trading_bot.trade_manager import all_open_trades
from trading_bot import liquidity_ws, data
from trading_bot.history import HISTORY_FILE, FIELDS as HISTORY_FIELDS

if Flask:
    app = Flask(__name__)

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
            qty = _coerce_float(trade.get("quantity", 0))
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
            row["current_price"] = current_price
            row["pnl_unrealized"] = pnl
            row["notional_value"] = abs(entry * qty)
            trades.append(row)
        return trades

    @app.route("/")
    def index():
        return render_template("index.html", current_year=datetime.utcnow().year)

    @app.route("/api/trades")
    def api_trades():
        """Return open trades with current price and unrealized PnL."""
        return jsonify(_trades_with_metrics())

    @app.route("/api/summary")
    def api_summary():
        """Aggregate metrics that power the dashboard widgets."""
        trades = _trades_with_metrics()
        total_positions = len(trades)
        total_exposure = sum(abs(t["quantity"]) for t in trades)
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
            entry["exposure"] += abs(trade["quantity"])
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
        app.run(host=host, port=port)
else:
    def start_dashboard(host: str, port: int):
        raise ImportError("Flask is required to run the dashboard")
