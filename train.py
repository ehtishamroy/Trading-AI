"""
Training Pipeline — Train all ML models with walk-forward validation.
Designed to run on vast.ai GPU or locally on your GTX 1070.

Uses expanding-window walk-forward CV with purge gaps to prevent data leakage.
Final models are trained on the largest available window and validated on held-out test set.

Usage:
  python train.py                    # Train for default market (EURUSD)
  python train.py --market XAUUSD    # Train for Gold
  python train.py --market all       # Train all markets
"""

import argparse
import numpy as np
import pandas as pd
import json
import torch
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import accuracy_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    MARKETS, ENTRY_TIMEFRAME,
    LSTM_SEQUENCE_LEN, MODELS_DIR, PREDICTION_HORIZON,
    TRIPLE_BARRIER_MAX_HOLDING,
)
from data.fetcher import load_data
from data.features import compute_all_features, get_feature_columns, normalize_features
from models.lstm_model import create_sequences, train_lstm
from models.xgboost_model import train_xgboost, get_feature_importance
import xgboost as xgb

# Purge gap: max_holding + 2 bars between train/test to prevent target leakage
PURGE_GAP = TRIPLE_BARRIER_MAX_HOLDING + 2


def _compute_sample_weights(n_samples: int, decay: float = 0.999) -> np.ndarray:
    """
    Exponential decay sample weights — recent data matters more.
    decay=0.999 means the oldest sample in 30k rows gets weight ~0.00005 of newest.
    """
    weights = np.array([decay ** (n_samples - 1 - i) for i in range(n_samples)])
    weights /= weights.sum()
    weights *= n_samples  # Scale so mean weight = 1.0
    return weights


def _walk_forward_splits(df: pd.DataFrame, n_folds: int = 5,
                         min_train_ratio: float = 0.4) -> list:
    """
    Generate expanding-window walk-forward splits with purge gaps.

    Returns list of (train_end_idx, test_start_idx, test_end_idx) index positions.
    Each fold: train on [0 : train_end], skip purge gap, test on [test_start : test_end].
    """
    n = len(df)
    test_size = int(n * (1 - min_train_ratio) / (n_folds + 1))
    min_train = int(n * min_train_ratio)

    splits = []
    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_start = train_end + PURGE_GAP
        test_end = min(test_start + test_size, n)

        if test_start >= n or test_end <= test_start:
            break
        splits.append((train_end, test_start, test_end))

    return splits


def train_market(market: str):
    """Train all models for a single market using walk-forward validation."""
    symbol = MARKETS[market]["mt5_symbol"]
    market_type = MARKETS[market].get("type", "forex")
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODELS FOR: {market}")
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
    df = compute_all_features(df, market_type=market_type)
    feature_cols = get_feature_columns(market_type=market_type)
    # Filter to only columns that actually exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = normalize_features(df, feature_cols)

    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    logger.info(f"Dataset: {len(df)} rows x {len(feature_cols)} features")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

    # ── Step 3: Walk-forward validation ──────────────────
    splits = _walk_forward_splits(df, n_folds=5)
    logger.info(f"Walk-forward: {len(splits)} folds, purge gap = {PURGE_GAP} bars")

    xgb_oos_scores = []
    lstm_oos_scores = []

    for fold_i, (train_end, test_start, test_end) in enumerate(splits):
        df_train = df.iloc[:train_end]
        df_test = df.iloc[test_start:test_end]
        logger.info(f"\n  Fold {fold_i+1}/{len(splits)}: "
                     f"Train {len(df_train)} bars | Test {len(df_test)} bars | "
                     f"Gap {test_start - train_end} bars")

        # XGBoost fold
        X_tr = df_train[feature_cols].values
        y_tr = df_train["target"].values
        X_te = df_test[feature_cols].values
        y_te = df_test["target"].values

        weights = _compute_sample_weights(len(y_tr))

        fold_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            early_stopping_rounds=20,
        )
        fold_model.fit(X_tr, y_tr, sample_weight=weights,
                       eval_set=[(X_te, y_te)], verbose=False)
        xgb_acc = accuracy_score(y_te, fold_model.predict(X_te))
        xgb_oos_scores.append(xgb_acc)
        logger.info(f"    XGBoost OOS accuracy: {xgb_acc:.2%}")

        # LSTM fold
        X_tr_seq, y_tr_seq = create_sequences(df_train, feature_cols)
        X_te_seq, y_te_seq = create_sequences(df_test, feature_cols)
        if len(X_tr_seq) > 100 and len(X_te_seq) > 0:
            fold_lstm = train_lstm(
                X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
                input_size=len(feature_cols),
                market=f"{market}_fold{fold_i}",
                epochs=50, batch_size=64,
            )
            fold_lstm.eval()
            fold_lstm.cpu()
            with torch.no_grad():
                preds = fold_lstm(torch.FloatTensor(X_te_seq))
                lstm_acc = ((preds > 0.5).float() == torch.FloatTensor(y_te_seq)).float().mean().item()
            lstm_oos_scores.append(lstm_acc)
            logger.info(f"    LSTM OOS accuracy: {lstm_acc:.2%}")

    if xgb_oos_scores:
        logger.info(f"\n  Walk-Forward XGBoost: mean={np.mean(xgb_oos_scores):.2%}, "
                     f"std={np.std(xgb_oos_scores):.2%}")
    if lstm_oos_scores:
        logger.info(f"  Walk-Forward LSTM: mean={np.mean(lstm_oos_scores):.2%}, "
                     f"std={np.std(lstm_oos_scores):.2%}")

    # ── Step 4: Train final models on full expanding window ──
    # Use last split config: train on 80%, test on final 20% with purge gap
    final_train_end = int(len(df) * 0.80)
    final_test_start = final_train_end + PURGE_GAP
    df_train = df.iloc[:final_train_end]
    df_test = df.iloc[final_test_start:]

    # Split train into train/val (90/10 of training set) for early stopping
    val_split = int(len(df_train) * 0.9)
    df_val = df_train.iloc[val_split:]
    df_train_final = df_train.iloc[:val_split]

    logger.info(f"\nFinal training: Train {len(df_train_final)} | Val {len(df_val)} | Test {len(df_test)}")

    # ── Final LSTM ───────────────────────────────────────
    logger.info("\nTraining final LSTM...")
    X_train_seq, y_train_seq = create_sequences(df_train_final, feature_cols)
    X_val_seq, y_val_seq = create_sequences(df_val, feature_cols)

    lstm_test_acc = None
    if len(X_train_seq) > 0:
        lstm_model = train_lstm(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            input_size=len(feature_cols),
            market=market,
        )

        X_test_seq, y_test_seq = create_sequences(df_test, feature_cols)
        if len(X_test_seq) > 0:
            lstm_model.eval()
            lstm_model.cpu()
            with torch.no_grad():
                test_pred = lstm_model(torch.FloatTensor(X_test_seq))
                lstm_test_acc = ((test_pred > 0.5).float() == torch.FloatTensor(y_test_seq)).float().mean().item()
                logger.info(f"LSTM Final Test Accuracy: {lstm_test_acc:.2%}")
    else:
        logger.warning("Not enough data for LSTM training")

    # ── Final XGBoost ────────────────────────────────────
    logger.info("\nTraining final XGBoost...")
    X_train_xgb = df_train_final[feature_cols].values
    y_train_xgb = df_train_final["target"].values
    X_val_xgb = df_val[feature_cols].values
    y_val_xgb = df_val["target"].values
    weights = _compute_sample_weights(len(y_train_xgb))

    xgb_model = train_xgboost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb,
                               market=market, sample_weight=weights)

    # Feature importance
    importance = get_feature_importance(xgb_model, feature_cols)
    logger.info(f"\nTop 10 features for {market}:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Test set evaluation
    X_test_xgb = df_test[feature_cols].values
    y_test_xgb = df_test["target"].values
    test_pred = xgb_model.predict(X_test_xgb)
    xgb_test_acc = accuracy_score(y_test_xgb, test_pred)
    logger.info(f"XGBoost Final Test Accuracy: {xgb_test_acc:.2%}")

    # ── Step 5: Write model metadata ─────────────────────
    data_range = f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "unknown"
    base_meta = {
        "market": market,
        "trained_at": datetime.now().isoformat(),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "data_range": data_range,
        "train_rows": len(df_train_final),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "purge_gap": PURGE_GAP,
        "walk_forward_folds": len(splits),
        "target_type": "triple_barrier",
    }

    combined_meta = {
        "lstm": {
            **base_meta,
            "model_name": f"lstm_{market}",
            "test_accuracy": lstm_test_acc,
            "wf_mean_accuracy": float(np.mean(lstm_oos_scores)) if lstm_oos_scores else None,
        },
        "xgboost": {
            **base_meta,
            "model_name": f"xgboost_{market}",
            "test_accuracy": xgb_test_acc,
            "wf_mean_accuracy": float(np.mean(xgb_oos_scores)) if xgb_oos_scores else None,
        },
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    existing = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing[market] = combined_meta
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    logger.info(f"Model metadata written -> {meta_path}")

    logger.success(f"\nAll models trained for {market}!")
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
        from data.mt5_connector import connect_mt5, disconnect_mt5
        from data.fetcher import fetch_and_save
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
