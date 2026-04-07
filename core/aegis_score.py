"""
Aegis Score Calculator — The single 0-100 composite score.
Combines ML confidence, regime fit, Claude verdict, and pattern history.
"""

import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import AEGIS_WEIGHTS, AEGIS_NO_TRADE, AEGIS_CAUTION


def calculate_aegis_score(
    ml_confidence: float,
    regime_fit: float,
    claude_confidence: int,
    pattern_match: float
) -> dict:
    """
    Calculate the Aegis Score (0-100).

    Args:
        ml_confidence: 0.0–1.0 from ensemble signal
        regime_fit: 0.0–1.0 (1.0 = signal fits regime perfectly)
        claude_confidence: 1–10 from Judge verdict
        pattern_match: 0.0–1.0 from pattern memory (historical win rate)

    Returns:
        {score: 0-100, level: 'RED'/'YELLOW'/'GREEN', can_trade: bool}
    """
    w = AEGIS_WEIGHTS

    # Normalize Claude confidence to 0-1 scale
    if claude_confidence is None:
        claude_confidence = 0
        
    claude_norm = claude_confidence / 10.0

    # Calculate weighted score
    raw_score = (
        ml_confidence * w["ml_confidence"] +
        regime_fit * w["regime_fit"] +
        claude_norm * w["claude_verdict"] +
        pattern_match * w["pattern_match"]
    )

    # Scale to 0-100
    score = round(raw_score * 100, 1)

    # Determine level
    if score >= AEGIS_CAUTION:
        level = "GREEN"
        label = "High Conviction ✅"
    elif score >= AEGIS_NO_TRADE:
        level = "YELLOW"
        label = "Proceed with Caution ⚠️"
    else:
        level = "RED"
        label = "No Trade ❌"

    can_trade = score >= AEGIS_NO_TRADE

    result = {
        "score": float(score),
        "level": level,
        "label": label,
        "can_trade": can_trade,
        "components": {
            "ml_confidence": float(round(ml_confidence * w["ml_confidence"] * 100, 1)),
            "regime_fit": float(round(regime_fit * w["regime_fit"] * 100, 1)),
            "claude_verdict": float(round(claude_norm * w["claude_verdict"] * 100, 1)),
            "pattern_match": float(round(pattern_match * w["pattern_match"] * 100, 1)),
        }
    }

    logger.info(f"Aegis Score: {score} ({level}) — {label}")
    return result


def format_aegis_display(aegis: dict) -> str:
    """Format Aegis Score for dashboard/telegram display."""
    comp = aegis["components"]
    bar = "█" * int(aegis["score"] / 5) + "░" * (20 - int(aegis["score"] / 5))

    return (
        f"═══ AEGIS SCORE: {aegis['score']}/100 ═══\n"
        f"[{bar}] {aegis['label']}\n\n"
        f"ML Confidence:  {comp['ml_confidence']:.1f}/35\n"
        f"Regime Fit:     {comp['regime_fit']:.1f}/30\n"
        f"Claude Verdict: {comp['claude_verdict']:.1f}/10\n"
        f"Pattern Match:  {comp['pattern_match']:.1f}/25\n"
    )
