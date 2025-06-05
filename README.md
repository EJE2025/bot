# Trading Bot

This repository contains a minimal crypto trading bot that retrieves market data from MEXC and executes orders on Bitget.
It now provides optional integration with additional exchanges and a small web
dashboard for monitoring open positions. Telegram and Discord notifications are
also available.

## Features

- Advanced technical indicators (RSI, MACD, ATR)
- Risk management parameters
- Stop-loss and take-profit enforcement with daily risk limit
- Symbol filtering and leverage setup
- Order execution through multiple exchanges using `ccxt`
- Advanced order types (market, limit and stop)
- Trade selection prefers the highest-probability signal with risk/reward >= 2:1
- Web dashboard at `http://localhost:8000` for real time monitoring
- Liquidity heatmap at `http://localhost:8001` via FastAPI
- Telegram/Discord notifications when trades are opened
- Modular architecture to ease future improvements
- Order book analysis for liquidity zones and heat map-based scoring
- Public MEXC endpoints for ticker, order book and kline data
- Real-time order book streaming via `wss://contract.mexc.com/edge`
- WebSocket listeners are started explicitly with `strategy.start_liquidity()`
  so importing the strategy does not trigger network connections
- Persistent trade history stored to `trade_history.csv`
- Optional machine learning models optimized with `trading_bot.optimizer`
- Train new ML models from CSVs using `python -m trading_bot.train_model`
- A mock exchange allows offline testing if Bitget is unreachable

## Usage

Install dependencies and run the bot:

```bash
pip install -r requirements.txt
python -m trading_bot.bot
python -m trading_bot.train_model mydata.csv --target result
uvicorn trading_bot.dashboard:app --reload
```

`get_market_data` fetches up to 500 candles by default using MEXC’s
`/api/v1/contract/kline` endpoint. You can adjust the `limit` parameter (1‑1000)
to load shorter or longer histories.
Downloaded candles are saved under `cache/` so repeated analyses can run
offline if the API becomes unavailable.

### Environment variables

API credentials and behavior are controlled via environment variables as
defined in `trading_bot/config.py`:

- `BITGET_API_KEY`, `BITGET_API_SECRET`, `BITGET_PASSPHRASE`
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `MEXC_API_KEY`, `MEXC_API_SECRET`
- `DEFAULT_EXCHANGE` (default `bitget`)
- `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID`
- `DISCORD_WEBHOOK`
- `MAX_OPEN_TRADES` (default 10)
- `DAILY_RISK_LIMIT` (default `-50`)
- `TEST_MODE` set to `1` to use a mock exchange without sending real orders
- `MODEL_PATH` path to saved ML model (default `model.pkl`)
- `STOP_ATR_MULT` ATR multiple for stop loss (default `1.5`)
- `WEBAPP_HOST` dashboard host (default `0.0.0.0`)
- `WEBAPP_PORT` dashboard port (default `8000`)


Copy `.env.example` to `.env` and fill in your API keys to get started.
