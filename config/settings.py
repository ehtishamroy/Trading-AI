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
    "XAUUSD": {"mt5_symbol": "XAUUSDm", "type": "commodity", "pip_value": 0.001},
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
MAX_RISK_PER_TRADE = 0.02             # 2% = $4 on $200
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
COMMISSION_RATE = 0.00007             # ~0.7 pips spread (Exness typical)
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-12-31"

# ─── AEGIS SCORE THRESHOLDS ──────────────────────────────
AEGIS_NO_TRADE = 55        # Below 55 = no trade (DEMO LEARNING MODE — was 65)
AEGIS_CAUTION = 70         # 55-70 = yellow (human must approve)  
# Above 70 = green (OK for auto-mode)

AEGIS_WEIGHTS = {
    "ml_confidence": 0.30,
    "sentiment": 0.15,
    "regime_fit": 0.20,
    "claude_verdict": 0.20,
    "pattern_match": 0.15,
}

# ─── API KEYS (from .env) ────────────────────────────────
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── MT5 SETTINGS ────────────────────────────────────────
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "Exness-MT5Trial")  # Demo server

# ─── CLAUDE AI ────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"      # Good balance of cost/quality
CLAUDE_MAX_TOKENS = 2000
CLAUDE_MIN_CONFIDENCE = 7             # Claude must score 7+/10

# ─── ML TRAINING ──────────────────────────────────────────
LSTM_SEQUENCE_LEN = 60                # 60 candles lookback
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
PREDICTION_HORIZON = 12              # Predict 12 bars ahead

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

# ─── TRADING SESSIONS (PKT timezone = UTC+5) ─────────────
TRADING_SESSIONS = {
    "EURUSD": {"start": "13:00", "end": "17:00", "tz": "Asia/Karachi"},  # London-NY overlap
    "XAUUSD": {"start": "18:00", "end": "23:00", "tz": "Asia/Karachi"},  # NY session
    "BTCUSD": {"start": "00:00", "end": "23:59", "tz": "Asia/Karachi"},  # 24/7
}
