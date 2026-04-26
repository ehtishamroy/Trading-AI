"""
Optuna Hyperparameter Optimization — Bayesian search for best model params.
Optimizes both XGBoost and LSTM using walk-forward profit factor as objective.

Usage:
  python optimize.py                     # Optimize for default market (EURUSD)
  python optimize.py --market XAUUSD     # Optimize for Gold
  python optimize.py --market all        # Optimize all markets
  python optimize.py --n-trials 100      # Run 100 Optuna trials (default: 50)
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import optuna
from pathlib import Path
from loguru import logger
from sklearn.metrics import accuracy_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    MARKETS, ENTRY_TIMEFRAME, MODELS_DIR,
    LSTM_SEQUENCE_LEN, TRIPLE_BARRIER_MAX_HOLDING,
)
from data.fetcher import load_data
from data.features import compute_all_features, get_feature_columns, normalize_features
from models.lstm_model import TradingLSTM, create_sequences
from models.xgboost_model import save_xgboost_model

# Purge gap between train/test
PURGE_GAP = TRIPLE_BARRIER_MAX_HOLDING + 2

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_walk_forward_data(df: pd.DataFrame, feature_cols: list, n_folds: int = 3):
    """
    Generate walk-forward fold data for optimization.
    Uses 3 folds (fewer than training) to keep optimization fast.
    """
    n = len(df)
    test_size = int(n * 0.6 / (n_folds + 1))
    min_train = int(n * 0.4)

    folds = []
    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_start = train_end + PURGE_GAP
        test_end = min(test_start + test_size, n)
        if test_start >= n or test_end <= test_start:
            break
        folds.append((
            df.iloc[:train_end],
            df.iloc[test_start:test_end],
        ))
    return folds


def _compute_sample_weights(n: int, decay: float = 0.999) -> np.ndarray:
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()
    weights *= n
    return weights


# ═══ XGBOOST OPTIMIZATION ═══════════════════════════════════════════

def _xgb_objective(trial, folds, feature_cols):
    """Optuna objective for XGBoost — maximize mean OOS accuracy across folds."""
    import xgboost as xgb

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 2.0),
    }

    scores = []
    for df_train, df_test in folds:
        X_tr = df_train[feature_cols].values
        y_tr = df_train["target"].values
        X_te = df_test[feature_cols].values
        y_te = df_test["target"].values
        weights = _compute_sample_weights(len(y_tr))

        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=20,
        )
        model.fit(X_tr, y_tr, sample_weight=weights,
                  eval_set=[(X_te, y_te)], verbose=False)
        acc = accuracy_score(y_te, model.predict(X_te))
        scores.append(acc)

    return float(np.mean(scores))


def optimize_xgboost(df: pd.DataFrame, feature_cols: list, market: str,
                     n_trials: int = 50) -> dict:
    """Run Optuna study for XGBoost and return best params."""
    folds = _get_walk_forward_data(df, feature_cols, n_folds=3)
    logger.info(f"Optimizing XGBoost for {market} ({n_trials} trials, {len(folds)} folds)...")

    study = optuna.create_study(direction="maximize", study_name=f"xgb_{market}")
    study.optimize(lambda trial: _xgb_objective(trial, folds, feature_cols),
                   n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.success(f"XGBoost best accuracy: {study.best_value:.2%}")
    logger.info(f"XGBoost best params: {best}")
    return best


# ═══ LSTM OPTIMIZATION ═══════════════════════════════════════════

def _lstm_objective(trial, folds, feature_cols):
    """Optuna objective for LSTM — maximize mean OOS accuracy across folds."""
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    seq_len = trial.suggest_int("seq_len", 15, 60, step=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = []

    for df_train, df_test in folds:
        X_tr, y_tr = create_sequences(df_train, feature_cols, seq_len=seq_len)
        X_te, y_te = create_sequences(df_test, feature_cols, seq_len=seq_len)

        if len(X_tr) < 100 or len(X_te) < 10:
            return 0.5  # Not enough data

        model = TradingLSTM(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).to(device)
        X_te_t = torch.FloatTensor(X_te).to(device)
        y_te_t = torch.FloatTensor(y_te).to(device)

        train_data = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Short training for HPO (30 epochs)
        best_val_loss = float("inf")
        patience = 0
        for epoch in range(30):
            model.train()
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                pred = model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_te_t), y_te_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break

            # Pruning: report intermediate value
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model.eval()
        with torch.no_grad():
            preds = model(X_te_t)
            acc = ((preds > 0.5).float() == y_te_t).float().mean().item()
        scores.append(acc)

    return float(np.mean(scores))


def optimize_lstm(df: pd.DataFrame, feature_cols: list, market: str,
                  n_trials: int = 30) -> dict:
    """Run Optuna study for LSTM and return best params."""
    folds = _get_walk_forward_data(df, feature_cols, n_folds=3)
    logger.info(f"Optimizing LSTM for {market} ({n_trials} trials, {len(folds)} folds)...")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"lstm_{market}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(lambda trial: _lstm_objective(trial, folds, feature_cols),
                   n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.success(f"LSTM best accuracy: {study.best_value:.2%}")
    logger.info(f"LSTM best params: {best}")
    return best


# ═══ MAIN ═══════════════════════════════════════════════════════

def optimize_market(market: str, n_trials_xgb: int = 50, n_trials_lstm: int = 30):
    """Run full Optuna optimization for a single market."""
    symbol = MARKETS[market]["mt5_symbol"]
    market_type = MARKETS[market].get("type", "forex")

    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZING HYPERPARAMETERS FOR: {market}")
    logger.info(f"{'='*60}")

    # Load and prepare data
    try:
        df = load_data(symbol, ENTRY_TIMEFRAME)
    except FileNotFoundError:
        logger.error(f"No data for {market}. Run data fetcher first!")
        return

    df = compute_all_features(df, market_type=market_type)
    feature_cols = get_feature_columns(market_type=market_type)
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = normalize_features(df, feature_cols)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    logger.info(f"Dataset: {len(df)} rows x {len(feature_cols)} features")

    # Optimize XGBoost
    xgb_params = optimize_xgboost(df, feature_cols, market, n_trials=n_trials_xgb)

    # Optimize LSTM
    lstm_params = optimize_lstm(df, feature_cols, market, n_trials=n_trials_lstm)

    # Save best params
    params_path = MODELS_DIR / f"best_params_{market}.json"
    best_params = {
        "market": market,
        "xgboost": xgb_params,
        "lstm": lstm_params,
    }
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.success(f"Best params saved -> {params_path}")

    return best_params


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--market", default="EURUSD",
                        help="Market to optimize: EURUSD, XAUUSD, BTCUSD, or 'all'")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials for XGBoost (LSTM uses 60% of this)")
    args = parser.parse_args()

    n_xgb = args.n_trials
    n_lstm = max(int(args.n_trials * 0.6), 15)

    if args.market == "all":
        for market in MARKETS:
            optimize_market(market, n_trials_xgb=n_xgb, n_trials_lstm=n_lstm)
    else:
        optimize_market(args.market, n_trials_xgb=n_xgb, n_trials_lstm=n_lstm)


if __name__ == "__main__":
    main()
