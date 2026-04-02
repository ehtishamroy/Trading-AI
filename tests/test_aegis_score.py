"""Tests for Aegis Score calculation."""

import pytest
from core.aegis_score import calculate_aegis_score


class TestAegisScore:
    def test_perfect_score(self):
        result = calculate_aegis_score(
            ml_confidence=1.0,
            sentiment_alignment=1.0,
            regime_fit=1.0,
            claude_confidence=10,
            pattern_match=1.0,
        )
        assert result["score"] == 100.0
        assert result["level"] == "GREEN"
        assert result["can_trade"] is True

    def test_zero_score(self):
        result = calculate_aegis_score(
            ml_confidence=0.0,
            sentiment_alignment=0.0,
            regime_fit=0.0,
            claude_confidence=0,
            pattern_match=0.0,
        )
        assert result["score"] == 0.0
        assert result["level"] == "RED"
        assert result["can_trade"] is False

    def test_threshold_boundary(self):
        # Score at exactly AEGIS_NO_TRADE should be tradeable
        from config.settings import AEGIS_NO_TRADE
        # Engineer inputs to get a score near the threshold
        result = calculate_aegis_score(
            ml_confidence=0.6,
            sentiment_alignment=0.5,
            regime_fit=0.6,
            claude_confidence=5,
            pattern_match=0.5,
        )
        if result["score"] >= AEGIS_NO_TRADE:
            assert result["can_trade"] is True
        else:
            assert result["can_trade"] is False

    def test_components_sum_to_score(self):
        result = calculate_aegis_score(
            ml_confidence=0.7,
            sentiment_alignment=0.6,
            regime_fit=0.8,
            claude_confidence=7,
            pattern_match=0.65,
        )
        comp = result["components"]
        component_sum = sum(comp.values())
        assert abs(component_sum - result["score"]) < 0.2  # Allow tiny float rounding

    def test_level_categories(self):
        # RED
        red = calculate_aegis_score(0.1, 0.1, 0.1, 1, 0.1)
        assert red["level"] == "RED"

        # GREEN
        green = calculate_aegis_score(1.0, 1.0, 1.0, 10, 1.0)
        assert green["level"] == "GREEN"

    def test_claude_confidence_normalized(self):
        # claude_confidence of 10 should contribute max to its weight
        high = calculate_aegis_score(0.5, 0.5, 0.5, 10, 0.5)
        low = calculate_aegis_score(0.5, 0.5, 0.5, 1, 0.5)
        assert high["score"] > low["score"]
        assert high["components"]["claude_verdict"] > low["components"]["claude_verdict"]
