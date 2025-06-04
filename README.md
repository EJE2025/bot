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
- Telegram/Discord notifications when trades are opened
- Modular architecture to ease future improvements
- Order book analysis for liquidity zones and heat map-based scoring
- Public MEXC endpoints for ticker, order book and kline data
- Persistent trade history stored to `trade_history.csv`
- Simple backtesting engine via `python -m trading_bot.backtest`
- Optional machine learning models optimized with `trading_bot.optimizer`
- Train new ML models from CSVs using `python -m trading_bot.train_model`

## Usage

Install dependencies and run the bot:

```bash
pip install -r requirements.txt
python -m trading_bot.bot
python -m trading_bot.train_model mydata.csv --target result
```

`get_market_data` fetches up to 500 candles by default using MEXC’s
`/api/v1/contract/kline` endpoint. You can adjust the `limit` parameter (1‑1000)
to load shorter or longer histories for backtesting.

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
- `MODEL_PATH` path to saved ML model (default `model.pkl`)
- `STOP_ATR_MULT` ATR multiple for stop loss (default `1.5`)
- `WEBAPP_HOST` dashboard host (default `0.0.0.0`)
- `WEBAPP_PORT` dashboard port (default `8000`)


Copy `.env.example` to `.env` and fill in your API keys to get started.
