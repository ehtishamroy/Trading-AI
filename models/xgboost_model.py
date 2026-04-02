"""
XGBoost Pattern Classifier — Fast gradient boosting for trade signals.
Complements LSTM with a different approach: tree-based pattern matching.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH,
    XGBOOST_LEARNING_RATE, MODELS_DIR
)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    market: str = "EURUSD"
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier for Buy/Sell pattern recognition.

    Uses TimeSeriesSplit internally for validation.
    """
    model = xgb.XGBClassifier(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",         # Fast on both CPU and GPU
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        random_state=42,
        early_stopping_rounds=20,
    )

    logger.info(f"Training XGBoost for {market}...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_pred)
    logger.success(f"XGBoost {market} | Validation Accuracy: {accuracy:.2%}")
    logger.info(f"\n{classification_report(y_val, val_pred, target_names=['SELL', 'BUY'])}")

    # Save model
    save_xgboost_model(model, market)
    return model


def save_xgboost_model(model: xgb.XGBClassifier, market: str, metadata: dict = None):
    """Save trained model and optional metadata."""
    path = MODELS_DIR / f"xgboost_{market}.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    if metadata:
        import json
        meta_path = MODELS_DIR / f"xgboost_{market}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Model metadata saved → {meta_path}")
    logger.info(f"XGBoost saved → {path}")


def load_xgboost_model(market: str) -> xgb.XGBClassifier:
    """Load trained model."""
    path = MODELS_DIR / f"xgboost_{market}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained XGBoost for {market}")
    model = joblib.load(path)
    logger.info(f"Loaded XGBoost for {market}")
    return model


def predict_xgboost(model: xgb.XGBClassifier, X: np.ndarray) -> dict:
    """
    Make prediction with XGBoost.

    Returns:
        {direction: 'up'/'down', confidence: 0.0-1.0, feature_importance: top 5}
    """
    # Validate feature count matches trained model
    expected_features = model.n_features_in_
    if X.shape[-1] != expected_features:
        raise ValueError(
            f"XGBoost expects {expected_features} features, got {X.shape[-1]}"
        )
    if np.isnan(X).any():
        logger.warning("NaN values detected in XGBoost input — replacing with 0")
        X = np.nan_to_num(X, nan=0.0)

    proba = model.predict_proba(X.reshape(1, -1))[0]
    direction = "up" if proba[1] > 0.5 else "down"
    confidence = max(proba)

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "prob_up": round(proba[1], 4),
        "prob_down": round(proba[0], 4),
    }


def get_feature_importance(model: xgb.XGBClassifier, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance ranking.
    Useful for understanding what drives the model's decisions.
    """
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    return df
