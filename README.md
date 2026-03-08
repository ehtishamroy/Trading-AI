# 🧠 ML Trading System v2 — with Claude AI Multi-Agent Debate

**Professional ML + AI trading system for Forex, Gold, and Bitcoin.**

Trade on Exness MT5 with machine learning models + Claude AI reasoning.

---

## 🏗️ Architecture

```
MT5 (Exness) → Data Pipeline → Feature Engine (60+ indicators)
                                        ↓
                    LSTM + XGBoost + Regime Detector (Ensemble)
                                        ↓
                    Claude AI Multi-Agent Debate (Bull vs Bear vs Judge)
                                        ↓
                    Aegis Score (0-100) → Risk Check → Trade Decision
                                        ↓
                    Telegram Alert + Dashboard → Human Approval → Execute
                                        ↓
                    Self-Learning Engine (Daily review + Weekly retrain)
```

## 📁 Project Structure

```
Trading AI/
├── config/
│   └── settings.py          # All configuration in one place
├── data/
│   ├── mt5_connector.py      # MT5 bridge (data + orders)
│   ├── fetcher.py            # Historical data download
│   ├── features.py           # 60+ technical indicators
│   └── news_sentiment.py     # News + sentiment scoring
├── models/
│   ├── lstm_model.py         # LSTM with attention (deep learning)
│   ├── xgboost_model.py      # XGBoost pattern classifier
│   ├── regime_detector.py    # Market regime detection
│   └── ensemble.py           # Signal combiner
├── core/
│   ├── claude_trader.py      # Multi-Agent Debate (Bull/Bear/Judge)
│   ├── risk_manager.py       # Position sizing + circuit breakers
│   ├── aegis_score.py        # Composite confidence score (0-100)
│   └── self_learning.py      # Trade journal + pattern memory
├── main.py                   # Main trading loop
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
├── .env.template             # API keys template
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
copy .env.template .env
# Edit .env with your MT5 login, Claude API key, etc.
```

### 3. Download Historical Data
```bash
python data/fetcher.py
```

### 4. Train Models
```bash
python train.py --market EURUSD --fetch
python train.py --market XAUUSD
python train.py --market BTCUSD
```

### 5. Start Trading (Demo)
```bash
python main.py
```

## ⚖️ Aegis Score

Every trade gets a composite confidence score (0-100):

| Component | Weight |
|-----------|--------|
| ML Model Confidence | 30% |
| Sentiment Alignment | 15% |
| Macro Regime Fit | 20% |
| Claude Judge Verdict | 20% |
| Historical Pattern Match | 15% |

- **< 65** = No Trade ❌
- **65-80** = Caution ⚠️ (human must approve)
- **80+** = Green ✅ (OK for auto-mode)

## ⚠️ Safety Rules (Non-Overridable)

1. Max 2% risk per trade
2. Max 3 consecutive losses → 4-hour pause
3. Max 5% daily drawdown → stop for the day
4. Max 15% total drawdown → system shuts off
5. Demo 30 days before live trading
