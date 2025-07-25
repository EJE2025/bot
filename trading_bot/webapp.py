try:
    from flask import Flask, render_template, jsonify
except ImportError:  # Flask not installed
    Flask = None

from trading_bot.trade_manager import all_open_trades
from trading_bot import liquidity_ws, data

if Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/trades")
    def api_trades():
        """Return open trades enriched with current price and unrealized PnL."""
        trades = []
        for t in all_open_trades():
            sym = t.get("symbol")
            entry = float(t.get("entry_price", 0))
            qty = float(t.get("quantity", 0))
            side = t.get("side", "BUY").upper()
            current_price = data.get_current_price_ticker(sym)
            if not current_price:
                current_price = entry
            pnl = (current_price - entry) * qty if side == "BUY" else (entry - current_price) * qty
            row = t.copy()
            row["current_price"] = current_price
            row["pnl_unrealized"] = pnl
            trades.append(row)
        return jsonify(trades)

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
        """Run the Flask dashboard in real-time with trades from trade_manager."""
        app.run(host=host, port=port)
else:
    def start_dashboard(host: str, port: int):
        raise ImportError("Flask is required to run the dashboard")

