"""
Pipeline Validation — Smoke-test the full ML pipeline end-to-end.
Loads data, computes features, generates signals, and runs backtest.
Must PASS Backtester.MINIMUM_REQUIREMENTS before any live trading.

Usage:
  python validate_pipeline.py                    # Validate EURUSD
  python validate_pipeline.py --market XAUUSD    # Validate Gold
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    MARKETS, ENTRY_TIMEFRAME, MODELS_DIR,
    LSTM_SEQUENCE_LEN, TRIPLE_BARRIER_MAX_HOLDING,
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR,
)
from data.fetcher import load_data
from data.features import compute_all_features, get_feature_columns, normalize_features
from models.lstm_model import load_lstm_model, create_sequences
from models.xgboost_model import load_xgboost_model
from models.regime_detector import RegimeDetector
from models.ensemble import combine_signals, StackingEnsemble
from backtesting.backtester import Backtester

PURGE_GAP = TRIPLE_BARRIER_MAX_HOLDING + 2


def validate_market(market: str):
    """Full pipeline validation for a single market."""
    symbol = MARKETS[market]["mt5_symbol"]
    market_type = MARKETS[market].get("type", "forex")

    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATING PIPELINE FOR: {market}")
    logger.info(f"{'='*60}")

    # ── Load & prepare data ──────────────────────────────
    try:
        df = load_data(symbol, ENTRY_TIMEFRAME)
    except FileNotFoundError:
        logger.error(f"No data for {market}. Run data fetcher first!")
        return False

    df = compute_all_features(df, market_type=market_type)
    feature_cols = get_feature_columns(market_type=market_type)
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_norm = normalize_features(df, feature_cols)
    df_norm = df_norm.dropna(subset=["target"])
    df_norm["target"] = df_norm["target"].astype(int)

    logger.info(f"Dataset: {len(df_norm)} rows x {len(feature_cols)} features")
    logger.info(f"Target: {df_norm['target'].value_counts().to_dict()}")

    # ── Hold-out test set (last 20%) ─────────────────────
    test_start = int(len(df_norm) * 0.80) + PURGE_GAP
    df_test = df_norm.iloc[test_start:]
    logger.info(f"Test set: {len(df_test)} rows ({df_test.index[0]} to {df_test.index[-1]})")

    # ── 1. Check model loads ─────────────────────────────
    logger.info("\n--- Model Load Check ---")
    lstm_ok = False
    xgb_ok = False

    try:
        lstm_model = load_lstm_model(market, len(feature_cols))
        lstm_ok = True
        logger.info(f"  LSTM: loaded OK")
    except Exception as e:
        logger.error(f"  LSTM: FAILED to load — {e}")

    try:
        xgb_model = load_xgboost_model(market)
        xgb_ok = True
        logger.info(f"  XGBoost: loaded OK")
        if hasattr(xgb_model, "n_features_in_"):
            if xgb_model.n_features_in_ != len(feature_cols):
                logger.warning(
                    f"  XGBoost expects {xgb_model.n_features_in_} features "
                    f"but pipeline produces {len(feature_cols)} — RETRAIN NEEDED"
                )
                xgb_ok = False
    except Exception as e:
        logger.error(f"  XGBoost: FAILED to load — {e}")

    if not (lstm_ok or xgb_ok):
        logger.error("No models available. Run: python train.py --market " + market)
        return False

    # ── 2. Accuracy on test set ──────────────────────────
    logger.info("\n--- Test Set Accuracy ---")
    y_true = df_test["target"].values

    if xgb_ok:
        X_test_xgb = df_test[feature_cols].values
        xgb_preds = xgb_model.predict(X_test_xgb)
        xgb_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
        xgb_acc = accuracy_score(y_true, xgb_preds)
        logger.info(f"  XGBoost accuracy: {xgb_acc:.2%}")
        logger.info(f"\n{classification_report(y_true, xgb_preds, target_names=['SL_HIT', 'TP_HIT'])}")

    if lstm_ok:
        X_test_seq, y_test_seq = create_sequences(df_test, feature_cols)
        if len(X_test_seq) > 0:
            lstm_model.eval()
            lstm_model.cpu()
            with torch.no_grad():
                lstm_proba = lstm_model(torch.FloatTensor(X_test_seq)).numpy()
                lstm_preds = (lstm_proba > 0.5).astype(int)
            lstm_acc = accuracy_score(y_test_seq, lstm_preds)
            logger.info(f"  LSTM accuracy: {lstm_acc:.2%}")

    # ── 3. ATR-based backtest ────────────────────────────
    logger.info("\n--- Backtest (ATR-based SL/TP) ---")

    # Generate signals from XGBoost on test set
    if xgb_ok:
        signals_list = []
        for i in range(len(df_test)):
            prob = xgb_proba[i] if i < len(xgb_proba) else 0.5
            if prob > 0.55:
                sig = 1   # Buy
            elif prob < 0.45:
                sig = -1  # Sell
            else:
                sig = 0   # Hold
            signals_list.append({"signal": sig, "confidence": abs(prob - 0.5) * 2})

        signals_df = pd.DataFrame(signals_list, index=df_test.index)

        # Use ATR for SL/TP instead of fixed percentages
        avg_atr_pct = df_test["atr_pct"].mean() if "atr_pct" in df_test.columns else 0.01
        sl_pct = avg_atr_pct * TRIPLE_BARRIER_SL_ATR
        tp_pct = avg_atr_pct * TRIPLE_BARRIER_TP_ATR

        bt = Backtester()
        metrics = bt.run_backtest(
            signals=signals_df,
            prices=df_test,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )

        # ── 4. Gate check ────────────────────────────────
        logger.info("\n--- Minimum Requirements Gate ---")
        gate = bt.passes_minimum(metrics)

        all_passed = gate.get("overall") == "PASSED ✅"
        for key, val in gate.items():
            if key == "overall":
                continue
            status = "PASS" if val["passed"] else "FAIL"
            logger.info(f"  [{status}] {key}: {val['value']} (threshold: {val['threshold']})")

        if all_passed:
            logger.success(f"\n{market} PASSES all gates — approved for demo trading")
        else:
            logger.warning(f"\n{market} FAILS gate check — DO NOT trade live")
            logger.info("Next steps: run optimize.py, retrain, then validate again")

        return all_passed

    return False


def main():
    parser = argparse.ArgumentParser(description="Validate ML pipeline")
    parser.add_argument("--market", default="EURUSD",
                        help="Market to validate: EURUSD, XAUUSD, BTCUSD, or 'all'")
    args = parser.parse_args()

    if args.market == "all":
        results = {}
        for market in MARKETS:
            results[market] = validate_market(market)
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION SUMMARY")
        for market, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {market}: {status}")
    else:
        validate_market(args.market)


if __name__ == "__main__":
    main()
