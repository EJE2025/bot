import os
from pathlib import Path
from typing import Tuple

from .secret_manager import get_secret


def _secret_or_env(name: str, default: str = "") -> str:
    """Return secrets from the secret manager falling back to environment vars."""

    secret_value = get_secret(name)
    if secret_value:
        return secret_value
    env_value = os.getenv(name)
    return env_value if env_value is not None else default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` between ``minimum`` and ``maximum``."""
    return max(minimum, min(maximum, value))


def _coerce_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, *, clamp: Tuple[float, float] | None = None) -> float:
    """Read a float from the environment with optional clamping."""
    value = _coerce_float(os.getenv(name), default)
    if clamp is not None:
        minimum, maximum = clamp
        value = _clamp(value, minimum, maximum)
    return value


def _positive_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Return a float constrained to be greater than ``minimum``."""
    value = _coerce_float(os.getenv(name), default)
    if value <= minimum:
        return default
    return value


def _int_env(name: str, default: int, *, clamp: Tuple[int, int] | None = None) -> int:
    """Return an integer from the environment respecting optional bounds."""

    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if clamp is not None:
        minimum, maximum = clamp
        value = max(minimum, min(maximum, value))
    return value


def _bool_env(name: str, default: bool) -> bool:
    """Return a boolean by interpreting common truthy strings."""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _parse_symbols(value: str) -> list[str]:
    symbols: list[str] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        token = token.upper().replace("-", "/").replace("_", "/")
        symbols.append(token)
    return symbols

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if load_dotenv is not None:
    load_dotenv(env_path)

_default_sqlite_path = Path(__file__).resolve().parent / "dashboard.sqlite3"
DATABASE_URL = _secret_or_env("DATABASE_URL", f"sqlite:///{_default_sqlite_path}")
DASHBOARD_SECRET_KEY = _secret_or_env("DASHBOARD_SECRET_KEY")
DASHBOARD_ADMIN_USERNAME = _str_env("DASHBOARD_ADMIN_USERNAME", "")
DASHBOARD_ADMIN_PASSWORD = _secret_or_env("DASHBOARD_ADMIN_PASSWORD")
DASHBOARD_ADMIN_PASSWORD_HASH = _secret_or_env("DASHBOARD_ADMIN_PASSWORD_HASH")
DASHBOARD_REQUIRE_AUTH = _bool_env("DASHBOARD_REQUIRE_AUTH", False)

BITGET_API_KEY = get_secret("BITGET_API_KEY") or ""
BITGET_API_SECRET = get_secret("BITGET_API_SECRET") or ""
BITGET_PASSPHRASE = get_secret("BITGET_PASSPHRASE") or os.getenv("BITGET_API_PASSPHRASE", "")

BINANCE_API_KEY = get_secret("BINANCE_API_KEY") or ""
BINANCE_API_SECRET = get_secret("BINANCE_API_SECRET") or ""

MEXC_API_KEY = get_secret("MEXC_API_KEY") or ""
MEXC_API_SECRET = get_secret("MEXC_API_SECRET") or ""

DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "bitget")

# Primary venue used for live trading and as default for data services.
PRIMARY_EXCHANGE = _str_env("PRIMARY_EXCHANGE", DEFAULT_EXCHANGE or "bitget")
# Exchange used by the historical/data fetcher (defaults to :data:`PRIMARY_EXCHANGE`).
DATA_EXCHANGE = _str_env("DATA_EXCHANGE", PRIMARY_EXCHANGE)
# Exchange powering websocket liquidity streams (defaults to :data:`PRIMARY_EXCHANGE`).
WS_EXCHANGE = _str_env("WS_EXCHANGE", PRIMARY_EXCHANGE)
# Redis endpoint for price/order event streaming.
REDIS_URL = _str_env("REDIS_URL", "redis://localhost:6379/0")
# Redis stream used by the streaming_service to fan-out market data.
MARKET_STREAM = _str_env("MARKET_STREAM", "market-data")
# Optional Redis stream used to fan-out trade lifecycle events.
TRADE_EVENT_STREAM = _str_env("TRADE_EVENT_STREAM", "trades")
# Consumer group name for price/event streams (auto-created when missing).
REDIS_CONSUMER_GROUP = _str_env("REDIS_CONSUMER_GROUP", "bot-consumers")
# Toggle Bitget connectivity (set to ``False`` when disabling any Bitget calls).
ENABLE_BITGET = _bool_env("ENABLE_BITGET", True)
# Toggle Binance connectivity (``False`` avoids reaching Binance endpoints).
ENABLE_BINANCE = _bool_env("ENABLE_BINANCE", False)
SYMBOLS = _parse_symbols(os.getenv("SYMBOLS", ""))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Dashboard & web UI ---------------------------------------------------------
# Default to the same origin as the Flask app unless explicitly configured to
# use the external FastAPI gateway. This keeps the standalone dashboard
# functional out of the box.
_default_gateway = os.getenv("GATEWAY_BASE_URL", "").strip().rstrip("/")
DASHBOARD_GATEWAY_BASE = (
    _str_env("DASHBOARD_GATEWAY_BASE", _default_gateway).strip().rstrip("/")
)
_socket_base = _str_env("DASHBOARD_SOCKET_BASE", "").strip()
DASHBOARD_SOCKET_BASE = _socket_base.rstrip("/") if _socket_base else ""
DASHBOARD_SOCKET_PATH = _str_env("DASHBOARD_SOCKET_PATH", "").strip()
_default_graphql = f"{DASHBOARD_GATEWAY_BASE}/graphql" if DASHBOARD_GATEWAY_BASE else ""
_default_ai_endpoint = (
    f"{DASHBOARD_GATEWAY_BASE}/ai/chat" if DASHBOARD_GATEWAY_BASE else ""
)
ANALYTICS_GRAPHQL_URL = _str_env("ANALYTICS_GRAPHQL_URL", _default_graphql).strip()
AI_ASSISTANT_URL = _str_env("AI_ASSISTANT_URL", _default_ai_endpoint).strip()
EXTERNAL_SERVICE_LINKS = _str_env("EXTERNAL_SERVICE_LINKS", "")

# Trading mode and permissions -------------------------------------------------
# Trading mode (``shadow`` performs simulated execution without touching the exchange).
BOT_MODE = os.getenv("BOT_MODE", "").strip().lower() or None
# Global flag to allow placing real orders on the exchange.
ENABLE_TRADING = _bool_env("ENABLE_TRADING", True)
ENABLE_MODEL = _bool_env("ENABLE_MODEL", True)
MAINTENANCE = False
# When ``True`` the bot avoids settlement (paper trading / dry-run mode).
DRY_RUN = _bool_env("DRY_RUN", False)
RUN_BACKTEST_ON_START = False
BACKTEST_CONFIG_PATH = os.getenv("BACKTEST_CONFIG_PATH")
BACKTEST_DATA_PATH = os.getenv("BACKTEST_DATA_PATH")

# ===== Order lifecycle & timeouts =====
PENDING_FILL_TIMEOUT_S = _int_env("PENDING_FILL_TIMEOUT_S", 30, clamp=(1, 600))
RECONCILE_INTERVAL_S = _int_env("RECONCILE_INTERVAL_S", 5, clamp=(1, 300))
# Intervalo principal del loop de trading en segundos.
LOOP_INTERVAL = _int_env("LOOP_INTERVAL", 60, clamp=(1, 3600))

# ===== WebSocket resilience =====
WS_PING_INTERVAL_S = _int_env("WS_PING_INTERVAL_S", 25, clamp=(5, 120))
WS_PONG_TIMEOUT_S = _int_env("WS_PONG_TIMEOUT_S", 10, clamp=(3, 60))
WS_BACKOFF_MIN_S = _int_env("WS_BACKOFF_MIN_S", 2, clamp=(1, 60))
WS_BACKOFF_MAX_S = _int_env("WS_BACKOFF_MAX_S", 60, clamp=(WS_BACKOFF_MIN_S, 600))

TRADING_MODE = os.getenv("TRADING_MODE", "live").strip().lower()
ALLOW_LIVE_TRADING = os.getenv("ALLOW_LIVE_TRADING", "1") == "1"
LIVE_TRADING_TOKEN_PATH = os.getenv("LIVE_TRADING_TOKEN_PATH", "")
LIVE_TRADING_TOKEN_VALUE = os.getenv(
    "LIVE_TRADING_TOKEN_VALUE", "ENABLE_LIVE_TRADING"
)

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
# Optional comma separated list of symbols to analyze when TEST_MODE is enabled
_test_syms = os.getenv("TEST_SYMBOLS", "")
TEST_SYMBOLS = [
    s.strip().upper().replace("/", "").replace("_", "")
    for s in _test_syms.split(",") if s.strip()
]
# Number of concurrent trades allowed (configurable via MAX_OPEN_TRADES env var)
try:
    MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "10"))
except (TypeError, ValueError):
    MAX_OPEN_TRADES = 10

MAX_CONCURRENT_POS = _int_env("MAX_CONCURRENT_POS", MAX_OPEN_TRADES, clamp=(1, 1000))
MAX_OPEN_TRADES = MAX_CONCURRENT_POS

# Maximum simultaneous trades per symbol
try:
    MAX_TRADES_PER_SYMBOL = int(os.getenv("MAX_TRADES_PER_SYMBOL", "1"))
except (TypeError, ValueError):
    MAX_TRADES_PER_SYMBOL = 1

MAX_POS_PER_SYMBOL = _int_env(
    "MAX_POS_PER_SYMBOL", MAX_TRADES_PER_SYMBOL, clamp=(1, MAX_CONCURRENT_POS)
)
MAX_TRADES_PER_SYMBOL = MAX_POS_PER_SYMBOL

# Cooldown in minutes between trades on the same symbol
try:
    COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "5"))
except (TypeError, ValueError):
    COOLDOWN_MINUTES = 5

# Seconds to wait before reopening a trade on the same symbol
TRADE_COOLDOWN = int(os.getenv("TRADE_COOLDOWN", str(COOLDOWN_MINUTES * 60)))
ENTRY_COOLDOWN_SECONDS = _int_env(
    "ENTRY_COOLDOWN_SECONDS", TRADE_COOLDOWN, clamp=(0, 24 * 60 * 60)
)
TRADE_COOLDOWN = ENTRY_COOLDOWN_SECONDS

# Maximum daily loss before trading stops
_default_daily_limit = abs(_coerce_float(os.getenv("DAILY_RISK_LIMIT"), -100.0))
if _default_daily_limit <= 0:
    _default_daily_limit = 100.0
MAX_DAILY_LOSS_USDT = _positive_float_env(
    "MAX_DAILY_LOSS_USDT", _default_daily_limit, minimum=0.0
)
DAILY_RISK_LIMIT = -abs(MAX_DAILY_LOSS_USDT)

AUTO_TRADE = _bool_env("AUTO_TRADE", True)
KILL_SWITCH_ON_DRIFT = _bool_env("KILL_SWITCH_ON_DRIFT", True)
DRIFT_PVALUE_CRIT = _float_env("DRIFT_PVALUE_CRIT", 0.01, clamp=(0.0, 1.0))
HIT_RATE_ROLLING_WARN = _float_env("HIT_RATE_ROLLING_WARN", 0.45, clamp=(0.0, 1.0))
HIT_RATE_ROLLING_CRIT = _float_env("HIT_RATE_ROLLING_CRIT", 0.40, clamp=(0.0, 1.0))
# Minimum notional enforced by the strategy regardless of exchange rules.
MIN_POSITION_SIZE_USDT = _positive_float_env(
    "MIN_POSITION_SIZE_USDT", 5.0, minimum=0.0
)
MAX_POSITION_SIZE_USDT = _positive_float_env(
    "MAX_POSITION_SIZE_USDT", 20.0, minimum=0.0
)
if MAX_POSITION_SIZE_USDT < MIN_POSITION_SIZE_USDT:
    MAX_POSITION_SIZE_USDT = MIN_POSITION_SIZE_USDT
# Use a constant position size expressed in USDT when ``True``.
USE_FIXED_POSITION_SIZE = _bool_env("USE_FIXED_POSITION_SIZE", False)
# Fixed position notional in USDT when :data:`USE_FIXED_POSITION_SIZE` is enabled.
FIXED_POSITION_SIZE_USDT = _positive_float_env(
    "FIXED_POSITION_SIZE_USDT", 10.0, minimum=0.0
)

# ===== Gestión dinámica de stop-loss =====
TRAILING_STOP_ENABLED = _bool_env("TRAILING_STOP_ENABLED", True)
TRAILING_STOP_TRIGGER = _float_env("TRAILING_STOP_TRIGGER", 0.02, clamp=(0.0, 1.0))
TRAILING_STOP_DISTANCE = _float_env("TRAILING_STOP_DISTANCE", 0.01, clamp=(0.0005, 0.5))
TRAILING_ATR_MULT = _positive_float_env("TRAILING_ATR_MULT", 3.0, minimum=0.0)
TRAILING_ATR_PERIOD = _int_env("TRAILING_ATR_PERIOD", 14, clamp=(1, 10_000))
# Whether to honour the exchange ``minNotional``/minimum order size constraints.
ENFORCE_EXCHANGE_MIN_NOTIONAL = _bool_env(
    "ENFORCE_EXCHANGE_MIN_NOTIONAL", True
)

BLACKLIST_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"}
UNSUPPORTED_SYMBOLS = {"AGIXTUSDT", "WHITEUSDT", "MAVIAUSDT", "PEPEUSDT"}

BASE_URL_MEXC = "https://contract.mexc.com/api/v1"
BASE_URL_BITGET = "https://api.bitget.com"
BASE_URL_BINANCE = "https://fapi.binance.com"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")

# Path to ML model used by the strategy
_MODEL_PATH_DEFAULT = "models/model.pkl"
MODEL_PATH = _str_env("MODEL_PATH", _MODEL_PATH_DEFAULT)
MODEL_DIR = _str_env("MODEL_DIR", str(Path(MODEL_PATH).parent))

# Path and parameters for sequential models (Keras/PyTorch)
MODEL_SEQ_PATH = _str_env("MODEL_SEQ_PATH", "")
MODEL_SEQ_WEIGHT = _float_env("MODEL_SEQ_WEIGHT", 0.2, clamp=(0.0, 1.0))
MODEL_SEQ_WINDOW = _int_env("MODEL_SEQ_WINDOW", 64, clamp=(4, 2048))
MODEL_SEQ_INTERVAL = _str_env("MODEL_SEQ_INTERVAL", "Min5")

_DATASET_DEFAULT = os.getenv("AUTO_TRAIN_DATA_PATH", "data/auto_train_data.csv")
DATASET_PATH = _str_env("DATASET_PATH", _DATASET_DEFAULT)
# Backwards compatibility: keep old name pointing to the same path
AUTO_TRAIN_DATA_PATH = DATASET_PATH

# Excel exports configuration -------------------------------------------------
EXPORTS_DIR = _str_env("EXPORTS_DIR", "./exports")
EXCEL_TRADES = _str_env("EXCEL_TRADES", "trades_closed.xlsx")
EXCEL_AI = _str_env("EXCEL_AI", "ai_status.xlsx")
EXCEL_OPS = _str_env("EXCEL_OPS", "ops.xlsx")

# Automatic retraining configuration
AUTO_TRAIN_ENABLED = _bool_env("AUTO_TRAIN_ENABLED", False)
AUTO_TRAIN_POLL_SECONDS = _int_env("AUTO_TRAIN_POLL_SECONDS", 60, clamp=(5, 3600))
RETRAIN_INTERVAL_TRADES = _int_env("RETRAIN_INTERVAL_TRADES", 50, clamp=(1, 1000000))
MIN_TRAIN_SAMPLE_SIZE = _int_env("MIN_TRAIN_SAMPLE_SIZE", 2000, clamp=(10, 1000000))
AUTO_TRAIN_MAX_SAMPLES = _int_env("AUTO_TRAIN_MAX_SAMPLES", 20000, clamp=(100, 1000000))
POST_DEPLOY_MIN_SAMPLES = _int_env("POST_DEPLOY_MIN_SAMPLES", 50, clamp=(1, 100000))
POST_DEPLOY_MIN_HIT_RATE = _float_env("POST_DEPLOY_MIN_HIT_RATE", 0.52, clamp=(0.0, 1.0))
POST_DEPLOY_MAX_DRIFT = _float_env("POST_DEPLOY_MAX_DRIFT", 0.08, clamp=(0.0, 1.0))

# Weight assigned to the predictive model when combining with heuristics
MODEL_WEIGHT = _float_env("MODEL_WEIGHT", 0.5, clamp=(0.0, 1.0))
# Minimum blended probability required to keep a signal
MIN_PROB_SUCCESS = _float_env("MIN_PROB_SUCCESS", 0.55, clamp=(0.0, 0.99))
_prob_threshold_raw = os.getenv("PROB_THRESHOLD")
if _prob_threshold_raw is not None:
    try:
        _prob_threshold_value = float(_prob_threshold_raw)
    except (TypeError, ValueError):
        _prob_threshold_value = MIN_PROB_SUCCESS
    _prob_threshold_value = _clamp(_prob_threshold_value, 0.0, 0.99)
else:
    _prob_threshold_value = None
PROB_THRESHOLD = (
    max(MIN_PROB_SUCCESS, _prob_threshold_value)
    if _prob_threshold_value is not None
    else None
)
# Additional margin applied over breakeven probability when filtering signals
PROBABILITY_MARGIN = _clamp(_float_env("PROBABILITY_MARGIN", 0.05), 0.0, 0.25)
FEE_AWARE_MARGIN_BPS = _int_env("FEE_AWARE_MARGIN_BPS", 2, clamp=(0, 1000))
# Estimated round-trip trading cost (fees + slippage) expressed as fraction of risk
FEE_EST = _positive_float_env("FEE_EST", 0.0006, minimum=0.0)
# Exchange fee rates (expressed as fraction of notional)
TAKER_FEE_RATE = _positive_float_env("TAKER_FEE_RATE", 0.0006, minimum=0.0)
MAKER_FEE_RATE = _positive_float_env("MAKER_FEE_RATE", 0.0002, minimum=0.0)

# Reinforcement learning policy tuning
RL_AGENT_ENABLED = _bool_env("RL_AGENT_ENABLED", False)
RL_ALGO = _str_env("RL_ALGO", "ppo").strip().lower() or "ppo"
RL_POLICY_PATH = _str_env("RL_POLICY_PATH", "models/rl_policy.zip").strip()
RL_BUFFER_CAPACITY = _int_env("RL_BUFFER_CAPACITY", 400, clamp=(10, 5000))
RL_LEARN_INTERVAL = _int_env("RL_LEARN_INTERVAL", 20, clamp=(1, 1000))
RL_MIN_TRAINING_SAMPLES = _int_env("RL_MIN_TRAINING_SAMPLES", 25, clamp=(1, 10000))
RL_LEARN_STEPS = _int_env("RL_LEARN_STEPS", 500, clamp=(10, 100000))
RL_TP_MULT_RANGE = (
    _float_env("RL_TP_MULT_MIN", 0.8, clamp=(0.1, 10.0)),
    _float_env("RL_TP_MULT_MAX", 2.5, clamp=(0.1, 20.0)),
)
RL_SL_MULT_RANGE = (
    _float_env("RL_SL_MULT_MIN", 0.8, clamp=(0.1, 10.0)),
    _float_env("RL_SL_MULT_MAX", 2.0, clamp=(0.1, 20.0)),
)
RL_DISCRETE_TP_BINS = _int_env("RL_DISCRETE_TP_BINS", 4, clamp=(2, 50))
RL_DISCRETE_SL_BINS = _int_env("RL_DISCRETE_SL_BINS", 3, clamp=(2, 50))
RL_PERSIST_AFTER_TRADE = _bool_env("RL_PERSIST_AFTER_TRADE", True)
RL_MAX_TP_PCT = _float_env("RL_MAX_TP_PCT", 0.5, clamp=(0.0, 10.0))
RL_MAX_STOP_LOSS_PCT = _float_env("RL_MAX_STOP_LOSS_PCT", 1.0, clamp=(0.0, 10.0))

# Noise filtering and volatility gating
NOISE_FILTER_METHOD = _str_env("NOISE_FILTER_METHOD", "ema").strip().lower() or "ema"
NOISE_FILTER_SPAN = _int_env("NOISE_FILTER_SPAN", 12, clamp=(1, 1000))
VOL_HIGH_TH = _float_env("VOL_HIGH_TH", 0.015, clamp=(0.0, 1.0))
VOL_MARGIN_BPS = _float_env("VOL_MARGIN_BPS", 10.0, clamp=(0.0, 1000.0))

SHADOW_MODE = _bool_env("SHADOW_MODE", False)
BACKTEST_REPORT_DIR = _str_env("BACKTEST_REPORT_DIR", "reports")
MAX_API_RETRIES = _int_env("MAX_API_RETRIES", 5, clamp=(0, 10))
API_BACKOFF_BASE = _float_env("API_BACKOFF_BASE", 0.2, clamp=(0.0, 5.0))
LATENCY_SLO_MS = _int_env("LATENCY_SLO_MS", 1000, clamp=(10, 10000))

# ATR multiple for stop loss calculation
STOP_ATR_MULT = float(os.getenv("STOP_ATR_MULT", "1.5"))
# Maximum relative loss (as fraction of entry price) allowed for stop loss levels
MAX_STOP_LOSS_PCT = _float_env("MAX_STOP_LOSS_PCT", 1.0, clamp=(0.0, 10.0))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = _float_env("RSI_OVERSOLD", 45.0, clamp=(1.0, 99.0))
RSI_OVERBOUGHT = _float_env("RSI_OVERBOUGHT", 55.0, clamp=(1.0, 99.0))
MIN_RISK_REWARD = float(
    os.getenv("MIN_RISK_REWARD", "1.3" if TEST_MODE else "2.0")
)
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "10"))

# Seconds to wait for a limit order to be filled before canceling
ORDER_FILL_TIMEOUT = int(os.getenv("ORDER_FILL_TIMEOUT", "15"))


# Seconds after which pending limit orders are cancelled automatically
ORDER_MAX_AGE = int(os.getenv("ORDER_MAX_AGE", "60"))

# Web dashboard configuration
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("WEBAPP_PORT", "8000"))
METRICS_PORT = _int_env("METRICS_PORT", 8001, clamp=(1024, 65535))


# Maximum acceptable slippage when closing a position
MAX_SLIPPAGE = float(os.getenv("MAX_SLIPPAGE", "0.01"))
CLOSE_REMAINING_TOLERANCE = _positive_float_env("CLOSE_REMAINING_TOLERANCE", 1e-6, minimum=0.0)

# Percent of available balance risked on each trade (e.g. 0.01 = 1%)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

# Minimum order size allowed when sizing positions
MIN_POSITION_SIZE = float(
    os.getenv("MIN_POSITION_SIZE", "1e-4" if TEST_MODE else "0.001")
)

# Enforce short/medium term operations by limiting how long a trade can stay open
try:
    MAX_TRADE_DURATION_MINUTES = int(
        os.getenv("MAX_TRADE_DURATION_MINUTES", "240")
    )
except (TypeError, ValueError):
    MAX_TRADE_DURATION_MINUTES = 240

# Optional in-memory trade history auditing
ENABLE_TRADE_HISTORY_LOG = os.getenv("ENABLE_TRADE_HISTORY_LOG", "0") == "1"
MAX_TRADE_HISTORY_SIZE = int(os.getenv("MAX_TRADE_HISTORY_SIZE", "1000"))
MAX_CLOSED_TRADES = _int_env("MAX_CLOSED_TRADES", 2000, clamp=(1, 1000000))
CLEAR_CLOSED_TRADES_AFTER_EXPORT = os.getenv(
    "CLEAR_CLOSED_TRADES_AFTER_EXPORT", "0"
) == "1"

# Prevent repeated entries by deduplicating similar signals for a short window
SIGNAL_IDEMPOTENCY_TTL = _int_env("SIGNAL_IDEMPOTENCY_TTL", 120, clamp=(0, 86400))

# Number of times to retry data fetch operations before falling back to cache
DATA_RETRY_ATTEMPTS = int(os.getenv("DATA_RETRY_ATTEMPTS", "3"))

# Maximum attempts when submitting or closing orders
ORDER_SUBMIT_ATTEMPTS = int(os.getenv("ORDER_SUBMIT_ATTEMPTS", "3"))


# Maximum number of simultaneous requests to exchanges
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

# Retry strategy for exchange API requests
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "4"))
API_RETRY_BACKOFF = _positive_float_env("API_RETRY_BACKOFF", 0.5, minimum=0.0)
API_RETRY_JITTER = _positive_float_env("API_RETRY_JITTER", 0.25, minimum=0.0)
API_MAX_RETRIES = _int_env("API_MAX_RETRIES", API_RETRY_ATTEMPTS, clamp=(0, 10))
API_RETRY_ATTEMPTS = API_MAX_RETRIES
API_BACKOFF_BASE_MS = _int_env(
    "API_BACKOFF_BASE_MS", int(API_RETRY_BACKOFF * 1000), clamp=(0, 60000)
)
if os.getenv("API_BACKOFF_BASE_MS") is not None:
    API_RETRY_BACKOFF = API_BACKOFF_BASE_MS / 1000.0

# Thresholds for system monitoring alerts
CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "0.8"))  # 80%
MEMORY_THRESHOLD_MB = float(os.getenv("MEMORY_THRESHOLD_MB", "500"))  # 500 MB
LATENCY_THRESHOLD_MS = float(os.getenv("LATENCY_THRESHOLD_MS", "1000"))  # 1 s

# Basic configuration sanity checks
if STOP_ATR_MULT <= 0:
    raise ValueError("STOP_ATR_MULT must be positive")
if RISK_PER_TRADE <= 0:
    raise ValueError("RISK_PER_TRADE must be positive")
if MODEL_WEIGHT < 0 or MODEL_WEIGHT > 1:
    raise ValueError("MODEL_WEIGHT must be within [0, 1]")
if MODEL_SEQ_WEIGHT < 0 or MODEL_SEQ_WEIGHT > 1:
    raise ValueError("MODEL_SEQ_WEIGHT must be within [0, 1]")
if MIN_PROB_SUCCESS < 0 or MIN_PROB_SUCCESS >= 1:
    raise ValueError("MIN_PROB_SUCCESS must be within [0, 1)")
# Model performance monitoring and drift handling
MODEL_PERFORMANCE_WINDOW = int(os.getenv("MODEL_PERFORMANCE_WINDOW", "50"))
MODEL_MIN_SAMPLES_FOR_MONITOR = int(os.getenv("MODEL_MIN_SAMPLES_FOR_MONITOR", "20"))
MODEL_MIN_WIN_RATE = _clamp(_float_env("MODEL_MIN_WIN_RATE", 0.52), 0.0, 1.0)
MODEL_MAX_CALIBRATION_DRIFT = _clamp(_float_env("MODEL_MAX_CALIBRATION_DRIFT", 0.08), 0.0, 0.5)
MODEL_WEIGHT_FLOOR = _clamp(_float_env("MODEL_WEIGHT_FLOOR", 0.1), 0.0, 1.0)
MODEL_WEIGHT_DEGRADATION = _clamp(_float_env("MODEL_WEIGHT_DEGRADATION", 0.5), 0.0, 1.0)

