from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from . import liquidity_ws

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    liquidity_ws.start(["BTC/USDT", "ETH/USDT"])

@app.get("/liquidity")
async def liquidity():
    return liquidity_ws.get_liquidity()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Liquidity Heatmap</title>
        <style>
            table { border-collapse: collapse; }
            th, td { border: 1px solid #ccc; padding: 4px; text-align: right; }
        </style>
    </head>
    <body>
        <h1>Liquidity Heatmap</h1>
        <table id="heat"></table>
        <script>
        async function load(){
            const resp = await fetch('/liquidity');
            const data = await resp.json();
            const table = document.getElementById('heat');
            table.innerHTML = '';
            for(const sym in data){
                const row = document.createElement('tr');
                const bids = data[sym].bids;
                const asks = data[sym].asks;
                const bidSum = Object.values(bids).reduce((a,b)=>a+parseFloat(b),0);
                const askSum = Object.values(asks).reduce((a,b)=>a+parseFloat(b),0);
                row.innerHTML = `<td>${sym}</td><td style='background:rgba(0,255,0,0.3)'>${bidSum.toFixed(2)}</td><td style='background:rgba(255,0,0,0.3)'>${askSum.toFixed(2)}</td>`;
                table.appendChild(row);
            }
        }
        setInterval(load, 2000);
        load();
        </script>
    </body>
    </html>
    """
