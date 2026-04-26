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
        self._prev_regime = None
        self._regime_streak = 0  # Consecutive bars of new regime needed before switch

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detect current market regime from feature DataFrame.
        Uses a rule-based approach with 20-bar exponential-weighted lookback
        and hysteresis (3 consecutive bars) to prevent noisy regime flipping.

        Returns:
            {regime: str, confidence: float, details: dict}
        """
        # Use 20-bar exponentially-weighted lookback for stability
        lookback = min(20, len(df))
        recent = df.iloc[-lookback:]

        # Exponential weights: most recent bar gets highest weight
        exp_weights = np.exp(np.linspace(-2, 0, lookback))
        exp_weights /= exp_weights.sum()

        def _ewm(series):
            """Exponentially-weighted mean over lookback window."""
            vals = series.values[-lookback:]
            if len(vals) < lookback:
                w = np.exp(np.linspace(-2, 0, len(vals)))
                w /= w.sum()
                return float(np.average(vals, weights=w))
            return float(np.average(vals, weights=exp_weights))

        # Extract key indicators with exponential weighting
        adx = _ewm(recent["adx"]) if "adx" in recent.columns else 20
        rsi = _ewm(recent["rsi_14"]) if "rsi_14" in recent.columns else 50
        bb_width = _ewm(recent["bb_width"]) if "bb_width" in recent.columns else 0.02
        atr_pct = _ewm(recent["atr_pct"]) if "atr_pct" in recent.columns else 0.01
        volatility_ratio = _ewm(recent["volatility_ratio"]) if "volatility_ratio" in recent.columns else 1.0
        price_vs_ema50 = _ewm(recent["price_vs_ema50"]) if "price_vs_ema50" in recent.columns else 0
        price_vs_ema200 = _ewm(recent["price_vs_ema200"]) if "price_vs_ema200" in recent.columns else 0
        ema_cross = recent["ema_cross_21_50"].iloc[-1] if "ema_cross_21_50" in recent.columns else 0
        vol_ratio = _ewm(recent["vol_ratio"]) if "vol_ratio" in recent.columns else 1.0
        atr_percentile = _ewm(recent["atr_percentile"]) if "atr_percentile" in recent.columns else 0.5

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
        if atr_percentile > 0.8:
            scores["HIGH_VOLATILITY"] += 2
        if vol_ratio > 2.0:
            scores["HIGH_VOLATILITY"] += 1
        if bb_width > 0.05:
            scores["HIGH_VOLATILITY"] += 1

        # Find raw winner
        raw_regime = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[raw_regime] / (total_score + 1e-9)

        # Hysteresis: require 3 consecutive bars of new regime before switching
        if raw_regime != self._prev_regime:
            self._regime_streak += 1
            if self._regime_streak >= 3:
                self._prev_regime = raw_regime
                self._regime_streak = 0
        else:
            self._regime_streak = 0

        # Use previous regime if streak not met (hysteresis)
        regime = self._prev_regime if self._prev_regime is not None else raw_regime

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "scores": scores,
            "raw_regime": raw_regime,
            "details": {
                "adx": round(adx, 2),
                "bb_width": round(bb_width, 4),
                "volatility_ratio": round(volatility_ratio, 2),
                "atr_pct": round(atr_pct, 4),
                "atr_percentile": round(atr_percentile, 2),
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
        raw = result.get("raw_regime", result["regime"])
        raw_note = f" (raw: {raw})" if raw != result["regime"] else ""
        return (
            f"## Market Regime\n"
            f"**Current**: {result['regime']}{raw_note} (confidence: {result['confidence']:.0%})\n"
            f"**Strategy**: {advice['strategy']}\n"
            f"**Bias**: {advice['bias']}\n"
            f"**Position Size**: {advice['position_multiplier']:.0%} of normal\n"
            f"**Stop Width**: {advice['stop_multiplier']}x ATR\n"
            f"\nDetails: ADX={result['details']['adx']}, "
            f"BB Width={result['details']['bb_width']}, "
            f"Vol Ratio={result['details']['volatility_ratio']}, "
            f"ATR Pctl={result['details'].get('atr_percentile', 'N/A')}"
        )
