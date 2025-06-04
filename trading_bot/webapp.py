from flask import Flask, render_template_string, jsonify

app = Flask(__name__)
_trades = []

PAGE = """<!doctype html>
<title>Trading Bot Dashboard</title>
<h1>Open Trades</h1>
<table border=1>
<tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>TP</th><th>SL</th></tr>
{% for t in trades %}
<tr><td>{{t['symbol']}}</td><td>{{t['side']}}</td><td>{{t['quantity']}}</td>
<td>{{'%.4f'%t['entry_price']}}</td><td>{{'%.4f'%t['take_profit']}}</td>
<td>{{'%.4f'%t['stop_loss']}}</td></tr>
{% endfor %}
</table>"""

@app.route("/")
def index():
    return render_template_string(PAGE, trades=_trades)

@app.route("/api/trades")
def api_trades():
    return jsonify(_trades)


def start_dashboard(trades, host="0.0.0.0", port=8000):
    global _trades
    _trades = trades
    app.run(host=host, port=port)
