"""
SHAP Feature Attribution — Explains WHY the ML model made each prediction.
Identifies which specific features caused a bad call.
Used in the self-learning loop to improve models.
"""

import shap
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import LOGS_DIR


def explain_xgboost_prediction(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list,
    top_n: int = 10,
) -> dict:
    """
    Explain a single XGBoost prediction using SHAP.

    Returns:
        {
            prediction: 'up'/'down',
            confidence: float,
            top_bullish_features: [{name, value, impact}],
            top_bearish_features: [{name, value, impact}],
            explanation: str (human-readable)
        }
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.reshape(1, -1))

    # For binary classification, shap_values is a single array
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Class 1 (UP)
    else:
        sv = shap_values[0]

    # Create feature impact table
    impacts = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "feature_value": X,
        "abs_impact": np.abs(sv),
    }).sort_values("abs_impact", ascending=False)

    # Top bullish features (SHAP > 0 = pushes toward UP)
    bullish = impacts[impacts["shap_value"] > 0].head(top_n)
    top_bull = [
        {"name": row["feature"], "value": round(row["feature_value"], 4),
         "impact": round(row["shap_value"], 4)}
        for _, row in bullish.iterrows()
    ]

    # Top bearish features (SHAP < 0 = pushes toward DOWN)
    bearish = impacts[impacts["shap_value"] < 0].head(top_n)
    top_bear = [
        {"name": row["feature"], "value": round(row["feature_value"], 4),
         "impact": round(row["shap_value"], 4)}
        for _, row in bearish.iterrows()
    ]

    # Prediction
    proba = model.predict_proba(X.reshape(1, -1))[0]
    pred = "up" if proba[1] > 0.5 else "down"
    conf = max(proba)

    # Human-readable explanation
    explanation = f"Model predicts {pred.upper()} ({conf:.0%} confident).\n"
    if top_bull:
        explanation += f"Main bullish driver: {top_bull[0]['name']} ({top_bull[0]['impact']:+.3f})\n"
    if top_bear:
        explanation += f"Main bearish driver: {top_bear[0]['name']} ({top_bear[0]['impact']:+.3f})\n"

    return {
        "prediction": pred,
        "confidence": round(conf, 4),
        "top_bullish_features": top_bull,
        "top_bearish_features": top_bear,
        "explanation": explanation,
        "all_impacts": impacts.to_dict("records"),
    }


def analyze_bad_trades(
    model: xgb.XGBClassifier,
    bad_trade_features: list[np.ndarray],
    feature_names: list,
) -> dict:
    """
    Analyze a batch of losing trades to find common patterns.
    Identifies which features consistently cause bad predictions.

    Returns:
        {recurring_culprits: [{feature, frequency, avg_impact}]}
    """
    if not bad_trade_features:
        return {"recurring_culprits": [], "message": "No bad trades to analyze"}

    explainer = shap.TreeExplainer(model)
    all_impacts = []

    for X in bad_trade_features:
        sv = explainer.shap_values(X.reshape(1, -1))
        if isinstance(sv, list):
            sv = sv[1][0]
        else:
            sv = sv[0]

        for i, name in enumerate(feature_names):
            all_impacts.append({"feature": name, "shap": sv[i]})

    df = pd.DataFrame(all_impacts)

    # Find features that consistently push wrong direction
    culprits = (
        df.groupby("feature")
        .agg(avg_impact=("shap", "mean"), abs_impact=("shap", lambda x: np.mean(np.abs(x))),
             count=("shap", "count"))
        .sort_values("abs_impact", ascending=False)
        .head(10)
    )

    result = [
        {"feature": name, "avg_impact": round(row["avg_impact"], 4),
         "abs_impact": round(row["abs_impact"], 4)}
        for name, row in culprits.iterrows()
    ]

    logger.info("SHAP Bad Trade Analysis:")
    for r in result[:5]:
        logger.info(f"  {r['feature']}: avg SHAP = {r['avg_impact']:+.4f}")

    return {"recurring_culprits": result}


def format_shap_for_claude(explanation: dict) -> str:
    """Format SHAP explanation for Claude's context."""
    lines = ["## SHAP Feature Attribution (Why This Signal?)"]
    lines.append(f"**{explanation['explanation']}**\n")

    if explanation["top_bullish_features"]:
        lines.append("**Bullish Drivers:**")
        for f in explanation["top_bullish_features"][:5]:
            lines.append(f"  📈 {f['name']}: {f['impact']:+.3f} (value: {f['value']:.3f})")

    if explanation["top_bearish_features"]:
        lines.append("\n**Bearish Drivers:**")
        for f in explanation["top_bearish_features"][:5]:
            lines.append(f"  📉 {f['name']}: {f['impact']:+.3f} (value: {f['value']:.3f})")

    return "\n".join(lines)
