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

TEST_MODE = False
MAX_OPEN_TRADES = 10
DAILY_RISK_LIMIT = -50.0

BLACKLIST_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"}
UNSUPPORTED_SYMBOLS = {"AGIXTUSDT", "WHITEUSDT", "MAVIAUSDT"}

BASE_URL_MEXC = "https://contract.mexc.com/api/v1"
BASE_URL_BITGET = "https://api.bitget.com"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
