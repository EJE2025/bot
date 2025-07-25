import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if load_dotenv is not None:
    load_dotenv(env_path)

BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", os.getenv("BITGET_API_PASSPHRASE", ""))

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")

DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "bitget")

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
# Number of concurrent trades allowed (configurable via MAX_OPEN_TRADES env var)
try:
    MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "10"))
except (TypeError, ValueError):
    MAX_OPEN_TRADES = 10
# Maximum daily loss before trading stops (configurable via DAILY_RISK_LIMIT env var)
try:
    DAILY_RISK_LIMIT = float(os.getenv("DAILY_RISK_LIMIT", "-50"))
except (TypeError, ValueError):
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

# Minimum order size allowed when sizing positions
MIN_POSITION_SIZE = float(os.getenv("MIN_POSITION_SIZE", "0.001"))

# Optional in-memory trade history auditing
ENABLE_TRADE_HISTORY_LOG = os.getenv("ENABLE_TRADE_HISTORY_LOG", "0") == "1"
MAX_TRADE_HISTORY_SIZE = int(os.getenv("MAX_TRADE_HISTORY_SIZE", "1000"))

