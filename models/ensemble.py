"""
Ensemble Signal Combiner — Stacking meta-learner + fallback weighted average.
Merges LSTM, XGBoost, and Regime Detector into a unified signal.

If a trained stacking model exists, uses LogisticRegression meta-learner.
Otherwise, falls back to calibrated weighted average.
"""

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from config.settings import ENSEMBLE_LSTM_WEIGHT, ENSEMBLE_XGBOOST_WEIGHT, MODELS_DIR
except ImportError:
    ENSEMBLE_LSTM_WEIGHT = 0.55
    ENSEMBLE_XGBOOST_WEIGHT = 0.45
    MODELS_DIR = Path("models/saved")


# ═══ REGIME ENCODING ═══════════════════════════════════════════

REGIME_MAP = {
    "TRENDING_UP": 0,
    "TRENDING_DOWN": 1,
    "RANGING": 2,
    "HIGH_VOLATILITY": 3,
}


def _build_meta_features(lstm_prob: float, xgb_prob: float, regime_name: str) -> np.ndarray:
    """
    Build feature vector for the stacking meta-learner.
    Features: [lstm_prob, xgb_prob, agreement, prob_diff, regime_encoded]
    """
    agreement = 1.0 if (lstm_prob > 0.5) == (xgb_prob > 0.5) else 0.0
    prob_diff = abs(lstm_prob - xgb_prob)
    regime_code = REGIME_MAP.get(regime_name, 2) / 3.0  # Normalize to [0, 1]
    return np.array([lstm_prob, xgb_prob, agreement, prob_diff, regime_code])


# ═══ STACKING META-LEARNER ═══════════════════════════════════

class StackingEnsemble:
    """
    Trained meta-learner that learns WHEN each base model is right.
    Uses LogisticRegression on [lstm_prob, xgb_prob, agreement, prob_diff, regime].
    """

    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.is_fitted = False

    def fit(self, lstm_probs: np.ndarray, xgb_probs: np.ndarray,
            regime_names: list, true_labels: np.ndarray):
        """
        Train the meta-learner on OOS predictions from base models.

        Args:
            lstm_probs: shape (n,) — LSTM P(TP hit) from walk-forward OOS
            xgb_probs:  shape (n,) — XGBoost P(TP hit) from walk-forward OOS
            regime_names: list of regime strings for each sample
            true_labels: shape (n,) — actual 0/1 labels
        """
        X = np.array([
            _build_meta_features(lp, xp, rn)
            for lp, xp, rn in zip(lstm_probs, xgb_probs, regime_names)
        ])
        self.meta_model.fit(X, true_labels)
        self.is_fitted = True
        logger.info("Stacking meta-learner fitted")

    def predict_proba(self, lstm_prob: float, xgb_prob: float, regime_name: str) -> float:
        """Return calibrated ensemble probability."""
        if not self.is_fitted:
            return None
        X = _build_meta_features(lstm_prob, xgb_prob, regime_name).reshape(1, -1)
        return float(self.meta_model.predict_proba(X)[0, 1])

    def save(self, market: str):
        path = MODELS_DIR / f"stacking_{market}.pkl"
        joblib.dump(self, path)
        logger.info(f"Stacking ensemble saved -> {path}")

    @staticmethod
    def load(market: str) -> "StackingEnsemble":
        path = MODELS_DIR / f"stacking_{market}.pkl"
        if not path.exists():
            return StackingEnsemble()
        se = joblib.load(path)
        logger.info(f"Loaded stacking ensemble for {market}")
        return se


# ═══ MAIN COMBINE FUNCTION ═══════════════════════════════════

def combine_signals(lstm_signal: dict, xgb_signal: dict, regime: dict,
                    market: str = None) -> dict:
    """
    Combine ML model outputs into one ensemble signal.

    Tries stacking meta-learner first. Falls back to weighted average if unavailable.

    Args:
        lstm_signal: {direction, confidence, raw_probability}
        xgb_signal:  {direction, confidence, prob_up}
        regime:      {regime: str, confidence: float}
        market:      Market name for loading stacking model (optional)

    Returns:
        {direction, confidence, signal_strength, agreement, combined_score, details}
    """
    # Validate inputs
    for name, sig in [("lstm", lstm_signal), ("xgboost", xgb_signal)]:
        conf = sig.get("confidence", 0)
        if not (0.0 <= conf <= 1.0):
            logger.warning(f"{name} confidence {conf} out of [0,1] — clamping")
            sig["confidence"] = max(0.0, min(1.0, conf))
        if sig.get("direction") not in ("up", "down", "neutral"):
            logger.warning(f"{name} direction '{sig.get('direction')}' invalid — defaulting to neutral")
            sig["direction"] = "neutral"

    # Extract raw probabilities
    lstm_prob = lstm_signal.get("raw_probability", 0.5 + (0.5 if lstm_signal["direction"] == "up" else -0.5) * lstm_signal["confidence"])
    xgb_prob = xgb_signal.get("prob_up", 0.5 + (0.5 if xgb_signal["direction"] == "up" else -0.5) * xgb_signal["confidence"])
    regime_name = regime.get("regime", "RANGING")

    # Try stacking meta-learner
    stacking_prob = None
    if market:
        try:
            stacker = StackingEnsemble.load(market)
            if stacker.is_fitted:
                stacking_prob = stacker.predict_proba(lstm_prob, xgb_prob, regime_name)
        except Exception as e:
            logger.debug(f"Stacking unavailable: {e}")

    if stacking_prob is not None:
        # Use stacking output
        combined_prob = stacking_prob
        method = "stacking"
    else:
        # Fallback: weighted average of raw probabilities
        combined_prob = lstm_prob * ENSEMBLE_LSTM_WEIGHT + xgb_prob * ENSEMBLE_XGBOOST_WEIGHT
        method = "weighted_avg"

    # Direction & confidence from combined probability
    if abs(combined_prob - 0.5) < 0.02:
        direction = "neutral"
    else:
        direction = "up" if combined_prob > 0.5 else "down"

    confidence = abs(combined_prob - 0.5) * 2.0  # Map [0.5, 1.0] -> [0.0, 1.0]

    # Agreement check
    agreement = lstm_signal["direction"] == xgb_signal["direction"]

    # Signal strength
    if confidence >= 0.50:
        strength = "STRONG"
    elif confidence >= 0.25:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    result = {
        "direction": direction,
        "confidence": round(confidence, 4),
        "signal_strength": strength,
        "agreement": agreement,
        "combined_score": round(combined_prob - 0.5, 4),
        "combined_probability": round(combined_prob, 4),
        "ensemble_method": method,
        "details": {
            "lstm": lstm_signal,
            "xgboost": xgb_signal,
            "regime": regime,
        }
    }

    logger.info(
        f"Ensemble ({method}): {direction.upper()} | "
        f"Confidence: {confidence:.2%} | "
        f"Strength: {strength} | "
        f"Agreement: {'YES' if agreement else 'NO'}"
    )

    return result


def format_for_claude(signal: dict) -> str:
    """Format ensemble signal as readable text for Claude."""
    lstm = signal["details"]["lstm"]
    xgb = signal["details"]["xgboost"]
    method = signal.get("ensemble_method", "weighted_avg")

    return (
        f"## ML Model Signals (Ensemble — {method})\n"
        f"**Combined Direction**: {signal['direction'].upper()} "
        f"({signal['signal_strength']}, {signal['confidence']:.0%} confidence)\n"
        f"**Model Agreement**: {'YES' if signal['agreement'] else 'NO'}\n\n"
        f"- LSTM (deep learning): {lstm['direction'].upper()} "
        f"({lstm['confidence']:.0%} confidence)\n"
        f"- XGBoost (pattern matching): {xgb['direction'].upper()} "
        f"({xgb['confidence']:.0%} confidence)"
    )
