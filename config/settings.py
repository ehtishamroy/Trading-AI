"""
Central Configuration — ALL settings in one place.
Never hardcode values in other files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ─── MARKET SETTINGS ─────────────────────────────────────
MARKETS = {
    "EURUSD": {"mt5_symbol": "EURUSDm", "type": "forex", "pip_value": 0.00001},
    "XAUUSD": {"mt5_symbol": "XAUUSDm", "type": "commodity", "pip_value": 0.01},
    "BTCUSD": {"mt5_symbol": "BTCUSDm", "type": "crypto", "pip_value": 0.01},
}
ACTIVE_MARKET = os.getenv("ACTIVE_MARKET", "EURUSD")  # Trade one at a time
ENTRY_TIMEFRAME = "M15"    # 15-minute candles for entries
TREND_TIMEFRAME = "H1"     # 1-hour candles for trend confirmation

# ─── ACCOUNT SETTINGS ────────────────────────────────────
STARTING_CAPITAL = 500.0              # USD (Exness demo)
LEVERAGE = 100                         # 1:100
LOT_SIZE = 0.01                        # Micro lots
TRADING_MODE = os.getenv("TRADING_MODE", "demo")  # 'demo' or 'live'

# ─── RISK MANAGEMENT ─────────────────────────────────────
MAX_RISK_PER_TRADE = 0.02             # 2% = $10 on $500
MAX_TRADES_PER_DAY = 4                # 3-4 quality trades
MAX_CONSECUTIVE_LOSSES = 3            # Pause after 3 losses in a row
PAUSE_AFTER_LOSSES_HOURS = 4          # Pause for 4 hours
MAX_DAILY_DRAWDOWN = 0.05             # Stop trading if day loss > 5%
MAX_TOTAL_DRAWDOWN = 0.15             # Shut off if total down > 15%
MIN_REWARD_RISK = 2.0                 # Minimum 2:1 reward/risk
ATR_STOP_MULTIPLIER = 1.5             # Stop = 1.5x ATR
MIN_SIGNAL_CONFIDENCE = 0.6           # Minimum ML confidence to consider

# ─── BACKTESTING ──────────────────────────────────────────
INITIAL_CAPITAL = STARTING_CAPITAL
COMMISSION_RATE = 0.00000             # Base commission separate from spread
SPREAD_MODEL = "fixed"                # Mode: "fixed" or "dynamic" (for data collection)
SPREAD_PIPS = 0.7                     # ~0.7 pips spread (Exness typical)
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-12-31"

# ─── AEGIS SCORE THRESHOLDS ──────────────────────────────
DEMO_MODE_OVERRIDES = True
if DEMO_MODE_OVERRIDES and TRADING_MODE == "demo":
    AEGIS_NO_TRADE = 55        # Below 55 = no trade (DEMO LEARNING MODE)
else:
    AEGIS_NO_TRADE = 65        # Production requirement

AEGIS_CAUTION = 70             # yellow zone (human must approve)  

AEGIS_WEIGHTS = {
    "ml_confidence": 0.30,
    "sentiment": 0.15,
    "regime_fit": 0.25,
    "claude_verdict": 0.10,
    "pattern_match": 0.20,
}

# ─── API KEYS (from .env) ────────────────────────────────
CLAUDE_API_KEY = ""  # Replaced by Ollama — no API key needed
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── PROXY SETTINGS ─────────────────────────────────────
PROXY_URL = os.getenv("PROXY_URL", "")  # e.g. socks5h://user:pass@host:port

# ─── MT5 SETTINGS ────────────────────────────────────────
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "Exness-MT5Trial")  # Demo server

# ─── OLLAMA LOCAL LLM (FREE — replaces Claude API) ──────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # Run: ollama pull llama3.1:8b
OLLAMA_MAX_TOKENS = 2000
CLAUDE_MIN_CONFIDENCE = 7             # Minimum LLM confidence to trade

# Legacy aliases (kept for any remaining references)
CLAUDE_MODEL = OLLAMA_MODEL
CLAUDE_MAX_TOKENS = OLLAMA_MAX_TOKENS

# ─── ML TRAINING ──────────────────────────────────────────
LSTM_SEQUENCE_LEN = 30               # 60 candles lookback
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
PREDICTION_HORIZON = 4              # Predict 12 bars ahead

XGBOOST_N_ESTIMATORS = 300
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.05

TRAIN_TEST_SPLIT = 0.7
TRAINING_DEVICE = "cuda"              # 'cuda' on vast.ai, 'cpu' on local

# ─── FILE PATHS ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
LOGS_DIR = PROJECT_ROOT / "logs"
JOURNAL_DIR = PROJECT_ROOT / "logs" / "journal"
MEMORY_DIR = PROJECT_ROOT / "logs" / "memory"

# Create directories
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, JOURNAL_DIR, MEMORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── ENSEMBLE WEIGHTS ────────────────────────────────────
# LSTM gets slightly more weight — captures temporal patterns better
ENSEMBLE_LSTM_WEIGHT = float(os.getenv("ENSEMBLE_LSTM_WEIGHT", "0.55"))
ENSEMBLE_XGBOOST_WEIGHT = float(os.getenv("ENSEMBLE_XGBOOST_WEIGHT", "0.45"))

# ─── TRADING SESSIONS (PKT timezone = UTC+5) ─────────────
TRADING_SESSIONS = {
    "EURUSD": {"start": "17:00", "end": "21:00", "tz": "Asia/Karachi"},  # London-NY overlap
    "XAUUSD": {"start": "17:00", "end": "21:00", "tz": "Asia/Karachi"},  # London-NY overlap
    "BTCUSD": {"start": "00:00", "end": "23:59", "tz": "Asia/Karachi"},  # 24/7
}


# ─── CONFIGURATION VALIDATION ────────────────────────────
import warnings

def _validate_config():
    """Validate critical configuration values on import."""
    if MAX_RISK_PER_TRADE <= 0 or MAX_RISK_PER_TRADE > 0.05:
        raise ValueError(f"MAX_RISK_PER_TRADE must be between 0 and 0.05, got {MAX_RISK_PER_TRADE}")
    if MAX_DAILY_DRAWDOWN <= 0:
        raise ValueError(f"MAX_DAILY_DRAWDOWN must be > 0, got {MAX_DAILY_DRAWDOWN}")
    if MAX_TOTAL_DRAWDOWN <= 0:
        raise ValueError(f"MAX_TOTAL_DRAWDOWN must be > 0, got {MAX_TOTAL_DRAWDOWN}")
    if LEVERAGE <= 0:
        raise ValueError(f"LEVERAGE must be > 0, got {LEVERAGE}")
    if abs(ENSEMBLE_LSTM_WEIGHT + ENSEMBLE_XGBOOST_WEIGHT - 1.0) > 0.01:
        warnings.warn(
            f"Ensemble weights should sum to 1.0, got "
            f"LSTM={ENSEMBLE_LSTM_WEIGHT} + XGB={ENSEMBLE_XGBOOST_WEIGHT} = "
            f"{ENSEMBLE_LSTM_WEIGHT + ENSEMBLE_XGBOOST_WEIGHT}"
        )
    if MT5_LOGIN == 0:
        warnings.warn("MT5_LOGIN is 0 — set MT5_LOGIN in .env for authenticated trading")
    if not MT5_PASSWORD:
        warnings.warn("MT5_PASSWORD is empty — set MT5_PASSWORD in .env")
    if not NEWS_API_KEY:
        warnings.warn("NEWS_API_KEY is empty — news sentiment will be unavailable")

_validate_config()
