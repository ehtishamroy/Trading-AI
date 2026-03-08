"""
Market Regime Detector — Classifies current market state.
Tells the system whether it's trending, ranging, or high-volatility.
Different regimes need different trading strategies.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import MODELS_DIR


class RegimeDetector:
    """
    Classifies market into regimes using volatility + trend features.

    Regimes:
    - TRENDING_UP:   Strong uptrend (ADX > 25, price above EMAs)
    - TRENDING_DOWN: Strong downtrend (ADX > 25, price below EMAs)
    - RANGING:       No clear direction (ADX < 20, tight Bollinger Bands)
    - HIGH_VOLATILITY: Big moves, wide ranges (VIX-like spikes)
    """

    REGIMES = {
        0: "TRENDING_UP",
        1: "TRENDING_DOWN",
        2: "RANGING",
        3: "HIGH_VOLATILITY",
    }

    def __init__(self):
        self.scaler = StandardScaler()

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detect current market regime from feature DataFrame.
        Uses a rule-based approach (more robust than ML for regime detection).

        Returns:
            {regime: str, confidence: float, details: dict}
        """
        latest = df.iloc[-1]

        # Extract key indicators
        adx = latest.get("adx", 20)
        rsi = latest.get("rsi_14", 50)
        bb_width = latest.get("bb_width", 0.02)
        atr_pct = latest.get("atr_pct", 0.01)
        volatility_ratio = latest.get("volatility_ratio", 1.0)
        price_vs_ema50 = latest.get("price_vs_ema50", 0)
        price_vs_ema200 = latest.get("price_vs_ema200", 0)
        ema_cross = latest.get("ema_cross_9_21", 0)
        vol_ratio = latest.get("vol_ratio", 1.0)

        # Score each regime
        scores = {
            "TRENDING_UP": 0,
            "TRENDING_DOWN": 0,
            "RANGING": 0,
            "HIGH_VOLATILITY": 0,
        }

        # --- Trend signals ---
        if adx > 25:
            if price_vs_ema50 > 0 and ema_cross > 0:
                scores["TRENDING_UP"] += 3
            elif price_vs_ema50 < 0 and ema_cross < 0:
                scores["TRENDING_DOWN"] += 3
        if adx > 35:
            if price_vs_ema200 > 0:
                scores["TRENDING_UP"] += 2
            else:
                scores["TRENDING_DOWN"] += 2

        # --- Ranging signals ---
        if adx < 20:
            scores["RANGING"] += 3
        if bb_width < 0.02:
            scores["RANGING"] += 2
        if 40 < rsi < 60:
            scores["RANGING"] += 1

        # --- High volatility signals ---
        if volatility_ratio > 1.5:
            scores["HIGH_VOLATILITY"] += 3
        if atr_pct > 0.02:
            scores["HIGH_VOLATILITY"] += 2
        if vol_ratio > 2.0:
            scores["HIGH_VOLATILITY"] += 1
        if bb_width > 0.05:
            scores["HIGH_VOLATILITY"] += 1

        # Find winner
        regime = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[regime] / (total_score + 1e-9)

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "scores": scores,
            "details": {
                "adx": round(adx, 2),
                "bb_width": round(bb_width, 4),
                "volatility_ratio": round(volatility_ratio, 2),
                "atr_pct": round(atr_pct, 4),
            }
        }

    def get_trading_advice(self, regime: str) -> dict:
        """
        Returns strategy adjustments based on regime.
        The system uses this to modify its behavior.
        """
        advice = {
            "TRENDING_UP": {
                "bias": "LONG",
                "strategy": "Trend following — buy dips to EMA",
                "position_multiplier": 1.0,   # Full size
                "stop_multiplier": 1.5,       # Normal stops
            },
            "TRENDING_DOWN": {
                "bias": "SHORT",
                "strategy": "Trend following — sell rallies to EMA",
                "position_multiplier": 1.0,
                "stop_multiplier": 1.5,
            },
            "RANGING": {
                "bias": "NEUTRAL",
                "strategy": "Mean reversion — buy support, sell resistance",
                "position_multiplier": 0.7,   # Reduced position size
                "stop_multiplier": 1.0,       # Tighter stops
            },
            "HIGH_VOLATILITY": {
                "bias": "CAUTIOUS",
                "strategy": "Reduced trading — only high conviction",
                "position_multiplier": 0.5,   # Half size
                "stop_multiplier": 2.0,       # Wider stops
            },
        }
        return advice.get(regime, advice["RANGING"])

    def format_for_claude(self, result: dict) -> str:
        """Format regime info for Claude's context."""
        advice = self.get_trading_advice(result["regime"])
        return (
            f"## Market Regime\n"
            f"**Current**: {result['regime']} (confidence: {result['confidence']:.0%})\n"
            f"**Strategy**: {advice['strategy']}\n"
            f"**Bias**: {advice['bias']}\n"
            f"**Position Size**: {advice['position_multiplier']:.0%} of normal\n"
            f"**Stop Width**: {advice['stop_multiplier']}x ATR\n"
            f"\nDetails: ADX={result['details']['adx']}, "
            f"BB Width={result['details']['bb_width']}, "
            f"Vol Ratio={result['details']['volatility_ratio']}"
        )
