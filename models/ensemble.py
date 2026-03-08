"""
Ensemble Signal Combiner — Merges signals from LSTM, XGBoost, and Regime Detector
into a single unified signal, ready for Claude's reasoning.
"""

import numpy as np
from loguru import logger


def combine_signals(lstm_signal: dict, xgb_signal: dict, regime: dict) -> dict:
    """
    Combine ML model outputs into one ensemble signal.

    Each model votes with its confidence. Regime adjusts the weight.

    Args:
        lstm_signal: {direction: 'up'/'down', confidence: 0.0-1.0}
        xgb_signal:  {direction: 'up'/'down', confidence: 0.0-1.0}
        regime:      {regime: str, confidence: float}

    Returns:
        {
            direction: 'up' / 'down' / 'neutral',
            confidence: 0.0-1.0,
            signal_strength: 'STRONG' / 'MODERATE' / 'WEAK',
            agreement: True/False (do both models agree?),
            details: {lstm details, xgb details, regime}
        }
    """
    # Convert directions to numeric: up = +1, down = -1
    lstm_vote = (1 if lstm_signal["direction"] == "up" else -1) * lstm_signal["confidence"]
    xgb_vote = (1 if xgb_signal["direction"] == "up" else -1) * xgb_signal["confidence"]

    # Weighted average (LSTM: 55%, XGBoost: 45%)
    # LSTM gets slightly more weight because it captures time patterns
    combined_score = (lstm_vote * 0.55) + (xgb_vote * 0.45)

    # Direction
    if abs(combined_score) < 0.1:
        direction = "neutral"
    else:
        direction = "up" if combined_score > 0 else "down"

    # Confidence = absolute value of combined score
    confidence = min(abs(combined_score) / 0.6, 1.0)  # Normalize to 0-1

    # Agreement check — both models pointing same way?
    agreement = lstm_signal["direction"] == xgb_signal["direction"]

    # Boost confidence if models agree, penalize if they disagree
    if agreement:
        confidence = min(confidence * 1.15, 1.0)
    else:
        confidence *= 0.75

    # Regime adjustment
    regime_name = regime.get("regime", "RANGING")
    if regime_name == "HIGH_VOLATILITY":
        confidence *= 0.8  # Lower confidence in volatile markets
    elif regime_name in ("TRENDING_UP", "TRENDING_DOWN"):
        # Boost if signal aligns with trend
        if (regime_name == "TRENDING_UP" and direction == "up") or \
           (regime_name == "TRENDING_DOWN" and direction == "down"):
            confidence = min(confidence * 1.1, 1.0)
        else:
            confidence *= 0.7  # Penalize counter-trend signals

    # Signal strength
    if confidence >= 0.75:
        strength = "STRONG"
    elif confidence >= 0.55:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    result = {
        "direction": direction,
        "confidence": round(confidence, 4),
        "signal_strength": strength,
        "agreement": agreement,
        "combined_score": round(combined_score, 4),
        "details": {
            "lstm": lstm_signal,
            "xgboost": xgb_signal,
            "regime": regime,
        }
    }

    logger.info(
        f"Ensemble: {direction.upper()} | "
        f"Confidence: {confidence:.2%} | "
        f"Strength: {strength} | "
        f"Agreement: {'YES' if agreement else 'NO'}"
    )

    return result


def format_for_claude(signal: dict) -> str:
    """Format ensemble signal as readable text for Claude."""
    lstm = signal["details"]["lstm"]
    xgb = signal["details"]["xgboost"]

    return (
        f"## ML Model Signals (Ensemble)\n"
        f"**Combined Direction**: {signal['direction'].upper()} "
        f"({signal['signal_strength']}, {signal['confidence']:.0%} confidence)\n"
        f"**Model Agreement**: {'YES ✅' if signal['agreement'] else 'NO ⚠️'}\n\n"
        f"- LSTM (deep learning): {lstm['direction'].upper()} "
        f"({lstm['confidence']:.0%} confidence)\n"
        f"- XGBoost (pattern matching): {xgb['direction'].upper()} "
        f"({xgb['confidence']:.0%} confidence)"
    )
