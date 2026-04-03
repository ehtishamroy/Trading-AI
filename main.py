"""
Main Trading Loop — Orchestrates the entire system.
Connects all components: Data → Features → ML → Claude → Risk → Execution.
"""

import time
import schedule
from datetime import datetime
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    ACTIVE_MARKET, MARKETS, ENTRY_TIMEFRAME, TREND_TIMEFRAME,
    TRADING_MODE, MIN_SIGNAL_CONFIDENCE, CLAUDE_MIN_CONFIDENCE,
    AEGIS_NO_TRADE, AEGIS_WEIGHTS, PROJECT_ROOT, LOGS_DIR,
)
from data.mt5_connector import connect_mt5, disconnect_mt5, get_ohlcv, get_current_price, get_account_info, place_order
from data.features import compute_all_features, get_feature_columns, normalize_features
from data.news_sentiment import get_sentiment_summary, format_for_claude
from models.regime_detector import RegimeDetector
from models.ensemble import combine_signals, format_for_claude as format_ensemble
from core.claude_trader import ClaudeTrader, build_market_context
from core.risk_manager import RiskManager
from core.aegis_score import calculate_aegis_score, format_aegis_display
from core.self_learning import TradingJournal


# ─── Setup ────────────────────────────────────────────────
logger.add(str(LOGS_DIR / "trading_{time}.log"), rotation="1 day", retention="30 days")

# Initialize all components
risk = RiskManager()
journal = TradingJournal()
regime_detector = RegimeDetector()
claude = ClaudeTrader()


def analyze_market(market: str = ACTIVE_MARKET) -> dict:
    """
    Run a complete market analysis — the core trading pipeline.

    Flow:
    1. Fetch latest data from MT5
    2. Compute features (60+ indicators)
    3. Get ML model predictions (LSTM + XGBoost)
    4. Detect market regime
    5. Get news sentiment
    6. Run Claude Multi-Agent Debate
    7. Calculate Aegis Score
    8. Risk check
    9. Return recommendation

    Returns:
        Full analysis dict with recommendation and reasoning
    """
    symbol = MARKETS[market]["mt5_symbol"]
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 ANALYZING: {market} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"{'='*60}")

    # ── Step 1: Fetch latest data ────────────────────────
    logger.info("Step 1: Fetching latest data from MT5...")
    df_15m = get_ohlcv(symbol, ENTRY_TIMEFRAME, 500)
    df_1h = get_ohlcv(symbol, TREND_TIMEFRAME, 200)

    if df_15m.empty:
        return {"error": "Cannot fetch market data. Is MT5 running?"}

    current = get_current_price(symbol)
    account = get_account_info()

    # ── Step 2: Compute features ─────────────────────────
    logger.info("Step 2: Computing features...")
    market_type = MARKETS[market].get("type", "forex")
    df_features = compute_all_features(df_15m, market_type=market_type)
    feature_cols = get_feature_columns()
    df_norm = normalize_features(df_features, feature_cols)

    # ── Step 3: ML predictions ───────────────────────────
    logger.info("Step 3: Getting ML predictions...")
    lstm_signal, xgb_signal = _get_ml_signals(market, df_norm, feature_cols)

    # ── Step 4: Regime detection ─────────────────────────
    logger.info("Step 4: Detecting market regime...")
    regime = regime_detector.detect(df_features)
    regime_text = regime_detector.format_for_claude(regime)
    regime_advice = regime_detector.get_trading_advice(regime["regime"])

    # ── Step 5: News sentiment ───────────────────────────
    logger.info("Step 5: Fetching news sentiment...")
    sentiment = get_sentiment_summary(market)
    sentiment_text = format_for_claude(sentiment, market)

    # ── Step 6: Ensemble signal ──────────────────────────
    logger.info("Step 6: Combining ML signals...")
    ensemble = combine_signals(lstm_signal, xgb_signal, regime)
    ensemble_text = format_ensemble(ensemble)

    # ── Step 7: Pattern memory ───────────────────────────
    pattern_text = journal.get_pattern_memory_for_claude(market, regime["regime"])

    # ── Step 7b: Reinforcement Learning (RL) Agent ───────
    logger.info("Step 7b: Querying RL Agent...")
    rl_decision_text = "RL Agent not trained/loaded yet."
    try:
        from pathlib import Path
        from stable_baselines3 import PPO
        from models.rl_env import RL_WINDOW_SIZE, RL_ADDITIONAL_STATES, RL_EXCLUDED_COLS
        rl_model_path = Path(f"models/saved/ppo_{market}.zip")
        if rl_model_path.exists():
            import numpy as np
            obs_cols = [c for c in df_features.columns if c not in RL_EXCLUDED_COLS and df_features[c].dtype != "object"]

            # Get last N rows (Frame Stacking) — uses shared constant
            obs_window = df_features.iloc[-RL_WINDOW_SIZE:][obs_cols].values.astype(np.float32)

            # Assume flat position: [position=0, unrealized_pnl=0]
            additional_states = np.zeros(RL_ADDITIONAL_STATES, dtype=np.float32)
            add_window = np.tile(additional_states, (RL_WINDOW_SIZE, 1))

            # Flatten to exactly match the TradingEnv observation space
            full_obs = np.concatenate((obs_window, add_window), axis=1)
            obs = full_obs.flatten().astype(np.float32)
            
            rl_model = PPO.load(str(rl_model_path))
            action, _states = rl_model.predict(obs, deterministic=True)
            
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            rl_action = action_map.get(int(action), "UNKNOWN")
            rl_decision_text = f"RL Agent Recommends: {rl_action}"
            logger.info(f"RL Agent output: {rl_action}")
    except Exception as e:
        logger.warning(f"Failed to query RL Agent: {e}")

    # ── Step 9: Pre-Score (Aegis without Claude) ─────────
    #   Calculate Aegis components BEFORE calling Claude.
    #   If ML+Sentiment+Regime+Pattern alone can't possibly
    #   reach the trade threshold even with a perfect Claude
    #   score, skip the 3x Claude API calls entirely.
    signal_dir = ensemble["direction"]
    sent_score = sentiment.get("score", 0)
    sentiment_alignment = 0.5  # Neutral default
    if (signal_dir == "up" and sent_score > 0) or (signal_dir == "down" and sent_score < 0):
        sentiment_alignment = 0.5 + abs(sent_score) * 0.5
    elif sent_score != 0:
        sentiment_alignment = 0.5 - abs(sent_score) * 0.3

    regime_fit = regime["confidence"]
    if regime_advice["bias"] == "CAUTIOUS":
        regime_fit *= 0.5

    # Load actual pattern win rate from journal history (0.5 fallback if no data)
    pattern_win_rate = journal.get_pattern_stats(market, regime["regime"], signal_dir)

    w = AEGIS_WEIGHTS
    pre_score = round((
        ensemble["confidence"] * w["ml_confidence"] +
        sentiment_alignment * w["sentiment"] +
        regime_fit * w["regime_fit"] +
        pattern_win_rate * w["pattern_match"]
    ) * 100, 1)

    # Claude's max contribution is now 10 pts (0.10 weight * 100). Only call if pre_score + 10 > AEGIS_NO_TRADE
    claude_max_pts = int(w["claude_verdict"] * 100)
    claude_call_threshold = AEGIS_NO_TRADE - claude_max_pts

    # ── Step 8: Claude Multi-Agent Debate ────────────────
    if pre_score < claude_call_threshold:
        logger.info(
            f"⏭️  Skipping Claude (pre-score {pre_score} < {claude_call_threshold}) — "
            f"signal too weak for Claude to change outcome. Saving API budget."
        )
        debate_result = {
            "verdict": {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": f"Pre-score {pre_score}/80 is too weak — skipped Claude to save API costs.",
            },
            "bull_case": "Skipped (pre-score too low)",
            "bear_case": "Skipped (pre-score too low)",
            "judge_raw": "Skipped",
        }
        verdict = debate_result["verdict"]
    else:
        logger.info(f"Step 8: Running Claude Multi-Agent Debate (pre-score {pre_score}/80 — Claude may tip the scale)...")
        market_context = build_market_context(
            market=market,
            current_price=current,
            ensemble_signal=ensemble_text,
            regime_info=regime_text,
            sentiment_info=sentiment_text,
            account_info=account,
            pattern_memory=pattern_text,
        )
        market_context += f"\n\n## Reinforcement Learning Agent (PPO)\n{rl_decision_text}\n"
        debate_result = claude.run_debate(market_context)
        verdict = debate_result["verdict"]

    # ── Step 9: Full Aegis Score (including Claude verdict) ─
    logger.info("Step 9: Calculating Full Aegis Score...")
    aegis = calculate_aegis_score(
        ml_confidence=ensemble["confidence"],
        sentiment_alignment=sentiment_alignment,
        regime_fit=regime_fit,
        claude_confidence=verdict.get("confidence", 0),
        pattern_match=pattern_win_rate,
    )

    # ── Step 10: Risk check ──────────────────────────────
    risk_check = risk.can_trade(account.get("balance", 200))

    # ── Compile result ───────────────────────────────────
    result = {
        "market": market,
        "timestamp": datetime.now().isoformat(),
        "current_price": current,
        "account": account,
        "ensemble": ensemble,
        "regime": regime,
        "sentiment": sentiment,
        "debate": debate_result,
        "verdict": verdict,
        "aegis": aegis,
        "risk_check": risk_check,
        "can_trade": aegis["can_trade"] and risk_check["allowed"],
        "decision": verdict.get("decision", "HOLD"),
    }

    # ── Display results ──────────────────────────────────
    logger.info(f"\n{format_aegis_display(aegis)}")
    logger.info(f"Decision: {verdict.get('decision', 'HOLD')}")
    logger.info(f"Reasoning: {verdict.get('reasoning', 'N/A')}")

    if not risk_check["allowed"]:
        logger.warning(f"🚫 BLOCKED by risk manager: {risk_check['reason']}")
    elif not aegis["can_trade"]:
        logger.info(f"⚠️ Aegis Score too low ({aegis['score']}) — skipping trade")
    else:
        # ── We have a valid trade! ──
        decision = verdict.get('decision', 'HOLD')
        sl = float(verdict.get("stop_loss") or 0.0)
        tp = float(verdict.get("take_profit") or 0.0)
        lot_size = risk_check.get("suggested_lot", 0.01)

        # Override HOLD when Aegis is strong (>=75) — ML stack is confident enough
        if decision == "HOLD" and aegis["score"] >= 75 and ensemble["direction"] in ("up", "down"):
            decision = "BUY" if ensemble["direction"] == "up" else "SELL"
            logger.info(
                f"⚡ Aegis GREEN ({aegis['score']}) overrides Claude HOLD → {decision} "
                f"(ML ensemble: {ensemble['direction']} @ {ensemble['confidence']:.0%})"
            )

        if decision == "HOLD":
            logger.info(f"✅ Aegis ({aegis['score']}) + Claude HOLD — skipping execution.")
        else:
            # Check dashboard state for Auto mode
            is_auto = False
            try:
                state_file = Path("logs/dashboard_state.json")
                if state_file.exists():
                    import json
                    with open(state_file) as f:
                        state = json.load(f)
                        is_auto = state.get("trading_mode") == "Auto"
            except Exception as e:
                logger.warning(f"Could not read dashboard state: {e}")

            if is_auto:
                logger.info("🤖 Auto-Mode Active: Executing trade on MT5...")

                mt5_symbol = MARKETS[market]["mt5_symbol"]
                order_res = place_order(
                    symbol=mt5_symbol,
                    order_type=decision,
                    lot_size=lot_size,
                    stop_loss=sl,
                    take_profit=tp,
                    comment=f"Aegis_{aegis['score']:.0f}"
                )

                if order_res.get("success"):
                    logger.success(f"✅ Trade Executed: {decision} {market} at {current.get('bid')} | SL: {sl} | TP: {tp}")
                    # Log to journal
                    journal.log_trade({
                        "ticket": order_res.get("ticket"),
                        "market": market,
                        "direction": decision,
                        "entry_price": order_res.get("price", current.get('bid')),
                        "lot_size": lot_size,
                        "aegis_score": aegis['score'],
                        "status": "OPEN",
                        "sl": sl,
                        "tp": tp
                    })
                else:
                    logger.error(f"❌ Trade Execution Failed: {order_res.get('error')}")

    # ── Send Telegram Alert (Always) ─────────────────────
    try:
        from core.telegram_notifier import send_telegram_alert
        
        # Determine status prefix
        decision = verdict.get("decision", "HOLD")
        if not risk_check["allowed"]:
            status_tag = f"🚫 BLOCKED: {risk_check['reason']}"
        elif not aegis["can_trade"]:
            status_tag = f"⚠️ HOLD (Score Too Low: {aegis['score']:.1f})"
        else:
            status_tag = f"🟢 TRADE ALERT: {decision}"
            if "is_auto" in locals() and is_auto:
                status_tag = f"[AUTO EXECUTED] {status_tag}"

        msg = (
            f"*{status_tag}*\n"
            f"Market: {market} @ {current.get('bid', 'N/A')}\n"
            f"Aegis Score: {aegis['score']:.1f}/100\n"
            f"ML Ensemble: {ensemble['confidence']*100:.1f}%\n"
            f"{rl_decision_text}\n\n"
            f"*Claude Judge Verdict*:\n{verdict.get('reasoning', '')[:500]}...\n\n"
            f"Dashboard: http://localhost:8502"
        )
        send_telegram_alert(msg)
    except Exception as e:
        logger.warning(f"Failed to send Telegram alert: {e}")

    # ── Save for dashboard ────────────────────────────────
    try:
        import json
        signal_file = Path("logs/latest_signal.json")
        signal_file.parent.mkdir(parents=True, exist_ok=True)
        # Make result JSON-serializable
        safe_result = json.loads(json.dumps(result, default=str))
        with open(signal_file, "w") as f:
            json.dump(safe_result, f, indent=2, default=str)
        logger.info(f"Signal saved for dashboard → {signal_file}")
    except Exception as e:
        logger.warning(f"Could not save signal: {e}")

    return result


def _get_ml_signals(market: str, df: object, feature_cols: list) -> tuple:
    """
    Get predictions from ML models (LSTM + XGBoost).
    If models aren't trained yet, returns placeholder signals.
    """
    try:
        from models.lstm_model import load_lstm_model, predict, create_sequences
        from config.settings import LSTM_SEQUENCE_LEN

        model = load_lstm_model(market, len(feature_cols))
        features = df[feature_cols].values
        if len(features) >= LSTM_SEQUENCE_LEN:
            latest_seq = features[-LSTM_SEQUENCE_LEN:]
            lstm_signal = predict(model, latest_seq)
        else:
            lstm_signal = {"direction": "neutral", "confidence": 0.5}
    except Exception as e:
        logger.warning(f"LSTM model unavailable for {market} — using placeholder: {e}")
        lstm_signal = {"direction": "neutral", "confidence": 0.5}

    try:
        from models.xgboost_model import load_xgboost_model, predict_xgboost
        model = load_xgboost_model(market)
        latest_features = df[feature_cols].values[-1]
        xgb_signal = predict_xgboost(model, latest_features)
    except Exception as e:
        logger.warning(f"XGBoost model unavailable for {market} — using placeholder: {e}")
        xgb_signal = {"direction": "neutral", "confidence": 0.5}

    return lstm_signal, xgb_signal


def run_preflight_checks() -> bool:
    """
    Startup pre-flight checks — verify all systems before entering trading loop.
    Returns True if all critical checks pass.
    """
    logger.info("Running pre-flight health checks...")
    all_ok = True

    # 1. Verify MT5 connection
    if not connect_mt5():
        logger.error("PREFLIGHT FAIL: Cannot connect to MT5")
        return False
    logger.info("  [OK] MT5 connection")

    # 2. Verify Ollama is running
    try:
        import requests
        from config.settings import OLLAMA_BASE_URL
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            logger.info("  [OK] Ollama reachable")
        else:
            logger.warning("  [WARN] Ollama returned non-200 — LLM debate may fail")
    except Exception:
        logger.warning("  [WARN] Ollama not reachable — LLM debate will be skipped")

    # 3. Verify saved models exist and load
    from config.settings import MODELS_DIR
    from data.features import get_feature_columns
    feature_cols = get_feature_columns()

    for model_type, ext in [("lstm", "pt"), ("xgboost", "pkl")]:
        model_path = MODELS_DIR / f"{model_type}_{ACTIVE_MARKET}.{ext}"
        if model_path.exists():
            try:
                if model_type == "lstm":
                    from models.lstm_model import load_lstm_model
                    m = load_lstm_model(ACTIVE_MARKET, len(feature_cols))
                else:
                    from models.xgboost_model import load_xgboost_model
                    m = load_xgboost_model(ACTIVE_MARKET)
                    # Verify feature count matches
                    if hasattr(m, "n_features_in_") and m.n_features_in_ != len(feature_cols):
                        logger.warning(
                            f"  [WARN] XGBoost expects {m.n_features_in_} features "
                            f"but feature pipeline produces {len(feature_cols)}"
                        )
                        all_ok = False
                logger.info(f"  [OK] {model_type.upper()} model loaded")
            except Exception as e:
                logger.warning(f"  [WARN] {model_type.upper()} model load failed: {e}")
        else:
            logger.warning(f"  [WARN] No {model_type.upper()} model for {ACTIVE_MARKET} — run train.py first")

    status = "ALL SYSTEMS GO" if all_ok else "SOME WARNINGS — proceeding with caution"
    logger.info(f"Pre-flight: {status}")
    return True  # Only return False for critical failures (MT5)


def run_trading_loop():
    """
    Main trading loop — runs every 15 minutes aligned with candle close.
    """
    logger.info(f"\n🚀 ML TRADING SYSTEM v2 STARTED")
    logger.info(f"Mode: {'🟡 DEMO' if TRADING_MODE == 'demo' else '🔴 LIVE'}")
    logger.info(f"Market: {ACTIVE_MARKET}")
    logger.info(f"Timeframe: {ENTRY_TIMEFRAME}")

    if not run_preflight_checks():
        logger.error("❌ Pre-flight checks failed. Exiting.")
        return

    # Run analysis immediately
    analysis = analyze_market()

    # Schedule to run every 15 minutes
    schedule.every(15).minutes.do(analyze_market)

    # Schedule daily reset
    schedule.every().day.at("00:00").do(risk.reset_daily)

    try:
        while True:
            schedule.run_pending()
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("\n🛑 Trading loop stopped by user")
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    run_trading_loop()
