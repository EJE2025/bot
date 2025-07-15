import os
from dotenv import load_dotenv

load_dotenv()

BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")

DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "bitget")

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
MAX_OPEN_TRADES = 10
DAILY_RISK_LIMIT = -50.0

BLACKLIST_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"}
UNSUPPORTED_SYMBOLS = {"AGIXTUSDT", "WHITEUSDT", "MAVIAUSDT"}

BASE_URL_MEXC = "https://contract.mexc.com/api/v1"
BASE_URL_BITGET = "https://api.bitget.com"
BASE_URL_BINANCE = "https://fapi.binance.com"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")

# Path to ML model used by the strategy
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

# ATR multiple for stop loss calculation
STOP_ATR_MULT = float(os.getenv("STOP_ATR_MULT", "1.5"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
MIN_RISK_REWARD = float(os.getenv("MIN_RISK_REWARD", "2.0"))
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "10"))

# Seconds to wait for a limit order to be filled before canceling
ORDER_FILL_TIMEOUT = int(os.getenv("ORDER_FILL_TIMEOUT", "15"))


# Seconds after which pending limit orders are cancelled automatically
ORDER_MAX_AGE = int(os.getenv("ORDER_MAX_AGE", "60"))

# Web dashboard configuration
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("WEBAPP_PORT", "8000"))


# Maximum acceptable slippage when closing a position
MAX_SLIPPAGE = float(os.getenv("MAX_SLIPPAGE", "0.01"))

# Percent of available balance risked on each trade (e.g. 0.01 = 1%)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

