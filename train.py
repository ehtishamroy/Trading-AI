"""
Training Pipeline — Train all ML models on historical data.
Designed to run on vast.ai GPU or locally on your GTX 1070.

Usage:
  python train.py                    # Train for default market (EURUSD)
  python train.py --market XAUUSD    # Train for Gold
  python train.py --market all       # Train all markets
"""

import argparse
import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    MARKETS, ENTRY_TIMEFRAME, TRAIN_TEST_SPLIT,
    LSTM_SEQUENCE_LEN, MODELS_DIR
)
from data.mt5_connector import connect_mt5, disconnect_mt5
from data.fetcher import fetch_and_save, load_data
from data.features import compute_all_features, get_feature_columns, normalize_features
from models.lstm_model import create_sequences, train_lstm
from models.xgboost_model import train_xgboost, get_feature_importance


def train_market(market: str):
    """Train all models for a single market."""
    symbol = MARKETS[market]["mt5_symbol"]
    logger.info(f"\n{'='*60}")
    logger.info(f"🧠 TRAINING MODELS FOR: {market}")
    logger.info(f"{'='*60}")

    # ── Step 1: Load data ────────────────────────────────
    logger.info("Loading data...")
    try:
        df = load_data(symbol, ENTRY_TIMEFRAME)
    except FileNotFoundError:
        logger.error(f"No data for {market}. Run data fetcher first!")
        logger.info("Run: python data/fetcher.py")
        return

    # ── Step 2: Feature engineering ──────────────────────
    logger.info("Computing features...")
    df = compute_all_features(df)
    feature_cols = get_feature_columns()
    df = normalize_features(df, feature_cols)

    logger.info(f"Dataset: {len(df)} rows × {len(feature_cols)} features")

    # ── Step 3: Train/test split (time-based, no shuffling) ──
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    val_start = int(split_idx + (len(df) - split_idx) * 0.5)

    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:val_start]
    df_test = df.iloc[val_start:]

    logger.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # ── Step 4: Train LSTM ───────────────────────────────
    logger.info("\n🔹 Training LSTM...")
    X_train_seq, y_train_seq = create_sequences(df_train, feature_cols)
    X_val_seq, y_val_seq = create_sequences(df_val, feature_cols)

    if len(X_train_seq) > 0:
        lstm_model = train_lstm(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            input_size=len(feature_cols),
            market=market
        )

        # Test set evaluation
        X_test_seq, y_test_seq = create_sequences(df_test, feature_cols)
        if len(X_test_seq) > 0:
            import torch
            lstm_model.eval()
            lstm_model.cpu()  # Move to CPU for evaluation
            with torch.no_grad():
                test_pred = lstm_model(torch.FloatTensor(X_test_seq))
                test_acc = ((test_pred > 0.5).float() == torch.FloatTensor(y_test_seq)).float().mean()
                logger.info(f"LSTM Test Accuracy: {test_acc:.2%}")
    else:
        logger.warning("Not enough data for LSTM training")

    # ── Step 5: Train XGBoost ────────────────────────────
    logger.info("\n🔹 Training XGBoost...")
    X_train_xgb = df_train[feature_cols].values
    y_train_xgb = df_train["target"].values
    X_val_xgb = df_val[feature_cols].values
    y_val_xgb = df_val["target"].values

    xgb_model = train_xgboost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, market=market)

    # Feature importance
    importance = get_feature_importance(xgb_model, feature_cols)
    logger.info(f"\nTop 10 features for {market}:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Test set evaluation
    from sklearn.metrics import accuracy_score
    X_test_xgb = df_test[feature_cols].values
    y_test_xgb = df_test["target"].values
    test_pred = xgb_model.predict(X_test_xgb)
    test_acc = accuracy_score(y_test_xgb, test_pred)
    logger.info(f"XGBoost Test Accuracy: {test_acc:.2%}")

    logger.success(f"\n✅ All models trained for {market}!")
    logger.info(f"Models saved in: {MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train ML trading models")
    parser.add_argument("--market", default="EURUSD",
                        help="Market to train: EURUSD, XAUUSD, BTCUSD, or 'all'")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch fresh data from MT5 before training")
    args = parser.parse_args()

    if args.fetch:
        logger.info("Fetching fresh data from MT5...")
        if connect_mt5():
            for market_key, info in MARKETS.items():
                if args.market == "all" or args.market == market_key:
                    fetch_and_save(info["mt5_symbol"], ENTRY_TIMEFRAME, 50000)
            disconnect_mt5()

    if args.market == "all":
        for market in MARKETS:
            train_market(market)
    else:
        train_market(args.market)


if __name__ == "__main__":
    main()
