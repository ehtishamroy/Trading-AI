"""
Weekly Retrain Pipeline — Automates model retraining on vast.ai.
Downloads latest data, retrains all models, validates against
current production models, and only deploys if new model is better.

Run manually: python retrain.py
Or schedule via cron: every Sunday at 3 AM
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    MARKETS, ENTRY_TIMEFRAME, MODELS_DIR, DATA_DIR, LOGS_DIR
)
from data.mt5_connector import connect_mt5, disconnect_mt5
from data.fetcher import fetch_and_save, load_data
from data.features import compute_all_features, get_feature_columns, normalize_features
from models.lstm_model import create_sequences, train_lstm
from models.xgboost_model import train_xgboost, get_feature_importance


# ─── Retrain Config ──────────────────────────────────────
RETRAIN_LOG = LOGS_DIR / "retrain_history.json"
BACKUP_DIR = MODELS_DIR / "backups"


def retrain_all(markets: list = None):
    """
    Full retrain pipeline for all markets.
    New model must beat old model on validation data to be deployed.
    """
    if markets is None:
        markets = list(MARKETS.keys())

    logger.info("=" * 60)
    logger.info(f"🔄 WEEKLY RETRAIN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)

    # Step 1: Fetch latest data
    logger.info("\n📥 Step 1: Fetching latest data from MT5...")
    if connect_mt5():
        for market in markets:
            symbol = MARKETS[market]["mt5_symbol"]
            fetch_and_save(symbol, ENTRY_TIMEFRAME, 50000)
        disconnect_mt5()
    else:
        logger.error("Cannot connect to MT5 — using existing data")

    results = {}
    for market in markets:
        result = retrain_market(market)
        results[market] = result

    # Save retrain log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    _save_retrain_log(log_entry)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RETRAIN SUMMARY")
    logger.info("=" * 60)
    for market, res in results.items():
        status = "✅ DEPLOYED" if res.get("deployed") else "❌ KEPT OLD"
        logger.info(f"  {market}: {status} — {res.get('reason', '')}")

    return results


def retrain_market(market: str) -> dict:
    """
    Retrain models for a single market.
    Compares new model vs old model on validation data.
    """
    symbol = MARKETS[market]["mt5_symbol"]
    logger.info(f"\n🧠 Retraining {market}...")

    # Load data
    try:
        df = load_data(symbol, ENTRY_TIMEFRAME)
    except FileNotFoundError:
        return {"deployed": False, "reason": "No data available"}

    # Feature engineering
    df = compute_all_features(df)
    feature_cols = get_feature_columns()
    df = normalize_features(df, feature_cols)

    # Time-based split
    split_70 = int(len(df) * 0.7)
    split_85 = int(len(df) * 0.85)

    df_train = df.iloc[:split_70]
    df_val = df.iloc[split_70:split_85]
    df_test = df.iloc[split_85:]

    logger.info(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # ── Train new LSTM ───────────────────────────────────
    X_train, y_train = create_sequences(df_train, feature_cols)
    X_val, y_val = create_sequences(df_val, feature_cols)
    X_test, y_test = create_sequences(df_test, feature_cols)

    new_lstm = None
    if len(X_train) > 100:
        new_lstm = train_lstm(
            X_train, y_train, X_val, y_val,
            input_size=len(feature_cols),
            market=f"{market}_candidate",  # Save as candidate, not production
        )

    # ── Train new XGBoost ────────────────────────────────
    X_train_xgb = df_train[feature_cols].values
    y_train_xgb = df_train["target"].values
    X_val_xgb = df_val[feature_cols].values
    y_val_xgb = df_val["target"].values
    X_test_xgb = df_test[feature_cols].values
    y_test_xgb = df_test["target"].values

    new_xgb = train_xgboost(
        X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb,
        market=f"{market}_candidate"
    )

    # ── Train new Reinforcement Learning (PPO) ───────────
    try:
        from retrain_rl import train_rl_agent
        logger.info(f"  🤖 Training RL Agent for {market} on 200,000 steps...")
        train_rl_agent(market, total_timesteps=200000, df=df)
        logger.success(f"  ✅ RL Agent training completed for {market}.")
    except Exception as e:
        logger.error(f"  ❌ RL Training failed for {market}: {e}")

    # ── Compare new vs old ───────────────────────────────
    new_score = _evaluate_model(new_xgb, X_test_xgb, y_test_xgb)
    old_score = _evaluate_old_model(market, X_test_xgb, y_test_xgb)

    logger.info(f"\n  New model accuracy: {new_score:.2%}")
    logger.info(f"  Old model accuracy: {old_score:.2%}")

    # ── Deploy only if better ────────────────────────────
    if new_score > old_score:
        _backup_old_model(market)
        _promote_candidate(market)
        logger.success(f"  ✅ New model DEPLOYED for {market} (improvement: {new_score - old_score:+.2%})")
        return {
            "deployed": True,
            "reason": f"New ({new_score:.2%}) > Old ({old_score:.2%})",
            "old_score": old_score,
            "new_score": new_score,
            "improvement": round(new_score - old_score, 4),
        }
    else:
        _cleanup_candidate(market)
        logger.info(f"  ❌ New model NOT better — keeping old for {market}")
        return {
            "deployed": False,
            "reason": f"New ({new_score:.2%}) <= Old ({old_score:.2%})",
            "old_score": old_score,
            "new_score": new_score,
        }


def _evaluate_model(model, X, y) -> float:
    """Evaluate XGBoost model accuracy."""
    from sklearn.metrics import accuracy_score
    pred = model.predict(X)
    return accuracy_score(y, pred)


def _evaluate_old_model(market: str, X, y) -> float:
    """Load and evaluate the current production model."""
    try:
        from models.xgboost_model import load_xgboost_model
        old_model = load_xgboost_model(market)
        return _evaluate_model(old_model, X, y)
    except FileNotFoundError:
        logger.info(f"  No existing model for {market} — any new model will be deployed")
        return 0.0


def _backup_old_model(market: str):
    """Backup current production model before replacing."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for ext in [".pkl", ".pt"]:
        for prefix in ["xgboost_", "lstm_"]:
            src = MODELS_DIR / f"{prefix}{market}{ext}"
            if src.exists():
                dst = BACKUP_DIR / f"{prefix}{market}_{timestamp}{ext}"
                shutil.copy2(src, dst)
                logger.info(f"  Backed up: {dst.name}")


def _promote_candidate(market: str):
    """Move candidate model to production."""
    for ext in [".pkl", ".pt"]:
        for prefix in ["xgboost_", "lstm_"]:
            candidate = MODELS_DIR / f"{prefix}{market}_candidate{ext}"
            production = MODELS_DIR / f"{prefix}{market}{ext}"
            if candidate.exists():
                shutil.move(str(candidate), str(production))


def _cleanup_candidate(market: str):
    """Remove rejected candidate model."""
    for ext in [".pkl", ".pt"]:
        for prefix in ["xgboost_", "lstm_"]:
            candidate = MODELS_DIR / f"{prefix}{market}_candidate{ext}"
            if candidate.exists():
                candidate.unlink()


def _save_retrain_log(entry: dict):
    """Append to retrain history log."""
    log = []
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG) as f:
            log = json.load(f)
    log.append(entry)
    with open(RETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weekly model retrain")
    parser.add_argument("--market", default="all", help="Market or 'all'")
    args = parser.parse_args()

    if args.market == "all":
        retrain_all()
    else:
        retrain_all([args.market])
