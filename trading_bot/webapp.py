"""Minimal Flask web application exposing trading state for the dashboard tests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from . import config, trade_manager, data

app = Flask(__name__)

HISTORY_FILE = Path("trade_history.csv")


def _load_history(max_rows: int = 50) -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []

    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) > max_rows:
        rows = rows[-max_rows:]

    rows = list(reversed(rows))
    return rows


@app.route("/api/trades")
def api_trades():
    return jsonify(trade_manager.all_open_trades())


@app.route("/api/summary")
def api_summary():
    open_trades = trade_manager.all_open_trades()
    closed_trades = trade_manager.all_closed_trades()
    per_symbol: dict[str, dict[str, Any]] = {}
    for t in open_trades:
        symbol = t.get("symbol") or ""
        entry = per_symbol.setdefault(symbol, {"symbol": symbol, "positions": 0})
        entry["positions"] += 1

    profits: list[float] = []
    for t in closed_trades:
        try:
            profit_val = float(t.get("profit") or t.get("realized_pnl") or 0.0)
        except (TypeError, ValueError):
            profit_val = 0.0
        profits.append(profit_val)

    wins = sum(1 for p in profits if p > 0)
    losses = sum(1 for p in profits if p < 0)
    realized_pnl = sum(profits)
    realized_balance = sum(p for p in profits if p > 0)
    win_rate = wins / len(profits) if profits else 0.0

    summary = {
        "total_positions": len(open_trades),
        "per_symbol": list(per_symbol.values()),
        "trading_active": bool(getattr(config, "ENABLE_TRADING", True)),
        "realized_pnl": realized_pnl,
        "realized_pnl_total": realized_pnl,
        "realized_balance": realized_balance,
        "win_rate": win_rate,
        "winning_positions": 0,
        "losing_positions": 0,
    }
    return jsonify(summary)


@app.route("/api/history")
def api_history():
    return jsonify(_load_history())


@app.route("/")
def index():
    api_base = getattr(config, "DASHBOARD_GATEWAY_BASE", "") or request.host_url.rstrip("/")
    analytics_graphql = getattr(config, "ANALYTICS_GRAPHQL_URL", "")
    ai_endpoint = getattr(config, "AI_ASSISTANT_URL", "")
    external_links = getattr(config, "EXTERNAL_SERVICE_LINKS", "")

    html = f"""
    <html>
      <head></head>
      <body
        data-api-base=\"{api_base}\"
        data-analytics-graphql=\"{analytics_graphql}\"
        data-ai-endpoint=\"{ai_endpoint}\"
        data-external-links=\"{external_links}\"
      >
      </body>
    </html>
    """
    return html


def start_dashboard(host: str, port: int) -> None:
    app.run(host=host, port=port)


__all__ = ["app", "HISTORY_FILE"]
