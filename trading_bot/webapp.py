from flask import Flask, render_template_string, jsonify
from trading_bot.trade_manager import all_open_trades

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Trading Bot Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap 5 CDN -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f5f6fa; }
        .table thead th { background-color: #222c3d; color: #fff; }
        .buy { color: #1abc9c; font-weight: bold; }
        .sell { color: #e74c3c; font-weight: bold; }
        .container { max-width: 900px; margin-top: 40px; }
        h1 { font-size: 2.5rem; margin-bottom: 1rem; color: #222c3d;}
        .card { box-shadow: 0 2px 12px 0 rgba(60,60,100,0.06); }
        .table { border-radius: 12px; overflow: hidden; }
        #splash {
            position: fixed;
            z-index: 9999;
            inset: 0;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: opacity 0.7s cubic-bezier(.7,.4,.6,1);
            opacity: 1;
        }
        #splash.hide {
            opacity: 0;
            pointer-events: none;
        }
        #splash-title {
            color: #fff;
            font-size: 6vw;
            font-weight: 900;
            letter-spacing: 0.04em;
            animation: zoomin 1s cubic-bezier(.4,.6,0.3,1.2);
            text-shadow: 0 0 40px #fff5, 0 4px 30px #0af4;
        }
        @keyframes zoomin {
            from { transform: scale(1.1); opacity: 1; }
            80%  { transform: scale(1.01); opacity: 1; }
            to   { transform: scale(1.17); opacity: 1; }
        }
        @media (max-width:600px){
            #splash-title{font-size:12vw;}
        }
    </style>
</head>
<body>
    <div id="splash">
        <div id="splash-title">Patatabot</div>
    </div>
    <div class="container" id="main-content" style="opacity:0; filter: blur(6px); transition: opacity .6s, filter .7s;">
        <div class="card p-4">
            <h1 class="text-center">Trading Bot Dashboard</h1>
            <table class="table table-hover table-bordered align-middle text-center mb-0">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>TP</th>
                        <th>SL</th>
                    </tr>
                </thead>
                <tbody id="ops">
                    <tr>
                        <td colspan="6" class="text-secondary py-4">Cargando operaciones...</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="text-center mt-4 text-muted small">

            &copy; 2025 Trading Bot Dashboard · Powered by Flask & Bootstrap


        </div>
    </div>
    <script>
        setTimeout(function() {
            document.getElementById('splash').classList.add('hide');
            document.getElementById('main-content').style.opacity = '1';
            document.getElementById('main-content').style.filter = 'blur(0)';
        }, 1100); // 1.1s para que la animación de zoom acabe antes de ocultar

        async function loadTrades() {
            try {
                let resp = await fetch('/api/trades');
                let trades = await resp.json();
                let html = '';
                if (trades && trades.length > 0) {
                    for (const t of trades) {
                        html += `<tr>
                            <td><span class="badge bg-primary">${t.symbol}</span></td>
                            <td class="${t.side && t.side.toLowerCase() === 'buy' ? 'buy' : 'sell'}">${t.side ? t.side.charAt(0).toUpperCase() + t.side.slice(1) : ''}</td>
                            <td>${t.quantity}</td>
                            <td>${parseFloat(t.entry_price).toFixed(4)}</td>
                            <td>${parseFloat(t.take_profit).toFixed(4)}</td>
                            <td>${parseFloat(t.stop_loss).toFixed(4)}</td>
                        </tr>`;
                    }
                } else {
                    html = `<tr><td colspan="6" class="text-secondary py-4">No hay operaciones abiertas.</td></tr>`;
                }
                document.getElementById('ops').innerHTML = html;
            } catch(e){
                document.getElementById('ops').innerHTML = `<tr><td colspan="6" class="text-danger py-4">Error cargando datos</td></tr>`;
            }
        }
        setInterval(loadTrades, 2500);
        loadTrades();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/api/trades")
def api_trades():
    return jsonify(all_open_trades())

def start_dashboard(host: str, port: int):
    """Run the Flask dashboard in real-time with trades from trade_manager."""
    app.run(host=host, port=port)

