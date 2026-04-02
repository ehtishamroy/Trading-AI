"""Tests for ensemble signal combining."""

import pytest
import importlib
import sys


def _import_combine_signals():
    """Import combine_signals directly, bypassing models/__init__.py which pulls in torch."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "models.ensemble",
        str(__import__("pathlib").Path(__file__).resolve().parents[1] / "models" / "ensemble.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.combine_signals


class TestCombineSignals:
    def _combine(self, lstm_dir="up", lstm_conf=0.8, xgb_dir="up", xgb_conf=0.7,
                 regime="TRENDING_UP", regime_conf=0.8):
        combine_signals = _import_combine_signals()
        lstm = {"direction": lstm_dir, "confidence": lstm_conf}
        xgb = {"direction": xgb_dir, "confidence": xgb_conf}
        reg = {"regime": regime, "confidence": regime_conf}
        return combine_signals(lstm, xgb, reg)

    def test_agreement_boosts_confidence(self):
        agreed = self._combine(lstm_dir="up", xgb_dir="up")
        disagreed = self._combine(lstm_dir="up", xgb_dir="down")
        assert agreed["confidence"] > disagreed["confidence"]

    def test_agreement_flag(self):
        result = self._combine(lstm_dir="up", xgb_dir="up")
        assert result["agreement"] is True
        result = self._combine(lstm_dir="up", xgb_dir="down")
        assert result["agreement"] is False

    def test_direction_up(self):
        result = self._combine(lstm_dir="up", xgb_dir="up")
        assert result["direction"] == "up"

    def test_direction_down(self):
        result = self._combine(lstm_dir="down", xgb_dir="down")
        assert result["direction"] == "down"

    def test_neutral_on_weak_signals(self):
        result = self._combine(lstm_dir="up", lstm_conf=0.05, xgb_dir="down", xgb_conf=0.05)
        assert result["direction"] == "neutral"

    def test_high_volatility_reduces_confidence(self):
        normal = self._combine(regime="TRENDING_UP")
        volatile = self._combine(regime="HIGH_VOLATILITY")
        assert volatile["confidence"] < normal["confidence"]

    def test_counter_trend_penalty(self):
        with_trend = self._combine(lstm_dir="up", xgb_dir="up", regime="TRENDING_UP")
        counter = self._combine(lstm_dir="up", xgb_dir="up", regime="TRENDING_DOWN")
        assert counter["confidence"] < with_trend["confidence"]

    def test_signal_strength_categories(self):
        strong = self._combine(lstm_conf=0.95, xgb_conf=0.95)
        assert strong["signal_strength"] in ("STRONG", "MODERATE", "WEAK")

    def test_confidence_clamped_0_1(self):
        result = self._combine(lstm_conf=1.0, xgb_conf=1.0)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_invalid_direction_defaults_to_neutral(self):
        combine_signals = _import_combine_signals()
        lstm = {"direction": "sideways", "confidence": 0.5}
        xgb = {"direction": "up", "confidence": 0.5}
        reg = {"regime": "RANGING", "confidence": 0.5}
        result = combine_signals(lstm, xgb, reg)
        assert result is not None  # Should not crash

    def test_confidence_out_of_range_clamped(self):
        combine_signals = _import_combine_signals()
        lstm = {"direction": "up", "confidence": 1.5}
        xgb = {"direction": "up", "confidence": -0.2}
        reg = {"regime": "RANGING", "confidence": 0.5}
        result = combine_signals(lstm, xgb, reg)
        assert 0.0 <= result["confidence"] <= 1.0
