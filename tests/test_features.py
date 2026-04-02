"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _make_ohlcv(n=500):
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=n, freq="15min")
    np.random.seed(42)
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)
    high = close + np.abs(np.random.randn(n) * 0.0003)
    low = close - np.abs(np.random.randn(n) * 0.0003)
    open_ = close + np.random.randn(n) * 0.0002
    volume = np.random.randint(100, 10000, n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


class TestComputeAllFeatures:
    def test_returns_dataframe(self):
        from data.features import compute_all_features
        df = _make_ohlcv()
        result = compute_all_features(df, market_type="forex")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_feature_count(self):
        from data.features import compute_all_features, get_feature_columns
        df = _make_ohlcv()
        result = compute_all_features(df, market_type="forex")
        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_no_nan_in_features(self):
        from data.features import compute_all_features, get_feature_columns
        df = _make_ohlcv()
        result = compute_all_features(df, market_type="forex")
        feature_cols = get_feature_columns()
        present_cols = [c for c in feature_cols if c in result.columns]
        nan_count = result[present_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in feature columns"

    def test_target_column_exists(self):
        from data.features import compute_all_features
        df = _make_ohlcv()
        result = compute_all_features(df, market_type="forex")
        assert "target" in result.columns

    def test_crypto_volume_features(self):
        from data.features import compute_all_features
        df = _make_ohlcv()
        result = compute_all_features(df, market_type="crypto")
        assert "obv" in result.columns
        # Crypto should have real volume features, not all zeros
        assert not (result["vol_ratio"] == 0).all()


class TestNormalization:
    def test_normalize_produces_no_nans(self):
        from data.features import compute_all_features, get_feature_columns, normalize_features
        df = _make_ohlcv()
        df = compute_all_features(df, market_type="forex")
        feature_cols = get_feature_columns()
        result = normalize_features(df, feature_cols)
        present_cols = [c for c in feature_cols if c in result.columns]
        nan_count = result[present_cols].isna().sum().sum()
        assert nan_count == 0

    def test_binary_cols_not_normalized(self):
        from data.features import compute_all_features, get_feature_columns, normalize_features
        df = _make_ohlcv()
        df = compute_all_features(df, market_type="forex")
        feature_cols = get_feature_columns()
        result = normalize_features(df, feature_cols)
        # Binary columns should only contain 0 or 1
        if "doji" in result.columns:
            assert set(result["doji"].unique()).issubset({0, 1, 0.0, 1.0})
