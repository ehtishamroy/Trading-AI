"""
Feature Engineering — Transforms raw OHLCV into 60+ ML-ready features.
Includes technical indicators, candlestick patterns, session context,
and target labels. Uses rolling normalization to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
import ta
from loguru import logger


def compute_all_features(df: pd.DataFrame, market_type: str = "forex") -> pd.DataFrame:
    """
    Compute ALL features from raw OHLCV data.
    Input: DataFrame with columns [open, high, low, close, volume]
    Output: Enriched DataFrame with 60+ indicator columns + target labels.
    """
    df = df.copy()
    eps = np.finfo(float).eps
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # ═══ TREND INDICATORS ═══════════════════════════════════
    df["ema_9"]   = ta.trend.ema_indicator(c, window=9)
    df["ema_21"]  = ta.trend.ema_indicator(c, window=21)
    df["ema_50"]  = ta.trend.ema_indicator(c, window=50)
    df["ema_200"] = ta.trend.ema_indicator(c, window=200)
    df["sma_20"]  = ta.trend.sma_indicator(c, window=20)

    # MACD
    macd = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # ADX (trend strength)
    adx = ta.trend.ADXIndicator(h, l, c)
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(h, l)
    df["ichi_a"]    = ichi.ichimoku_a()
    df["ichi_b"]    = ichi.ichimoku_b()
    df["ichi_base"] = ichi.ichimoku_base_line()
    df["ichi_conv"] = ichi.ichimoku_conversion_line()

    # ═══ MOMENTUM INDICATORS ════════════════════════════════
    df["rsi_14"]   = ta.momentum.rsi(c, window=14)
    df["rsi_7"]    = ta.momentum.rsi(c, window=7)

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"]  = stoch.stoch()
    df["stoch_d"]  = stoch.stoch_signal()

    df["cci"]       = ta.trend.cci(h, l, c, window=20)
    df["williams_r"] = ta.momentum.williams_r(h, l, c, lbp=14)
    df["roc_10"]    = ta.momentum.roc(c, window=10)
    df["mfi_14"]    = ta.volume.money_flow_index(h, l, c, v, window=14)

    # ═══ VOLATILITY INDICATORS ══════════════════════════════
    bb = ta.volatility.BollingerBands(c)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()

    df["atr_14"]   = ta.volatility.average_true_range(h, l, c, window=14)
    df["atr_pct"]  = df["atr_14"] / c

    kc = ta.volatility.KeltnerChannel(h, l, c)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()

    # ═══ VOLUME INDICATORS ══════════════════════════════════
    if market_type in ["crypto", "stock"]:
        df["obv"]        = ta.volume.on_balance_volume(c, v)
        df["cmf_20"]     = ta.volume.chaikin_money_flow(h, l, c, v, window=20)
        df["vol_sma_20"] = v.rolling(20).mean()
        df["vol_ratio"]  = v / (df["vol_sma_20"] + eps)
    else:
        # Forex/Commodities tick volume is unreliable — set to 0.0
        df["obv"]        = 0.0
        df["cmf_20"]     = 0.0
        df["vol_ratio"]  = 0.0

    # ═══ PRICE-DERIVED FEATURES ═════════════════════════════
    df["returns_1"]  = c.pct_change(1)
    df["returns_3"]  = c.pct_change(3)
    df["returns_7"]  = c.pct_change(7)
    df["returns_14"] = c.pct_change(14)
    df["log_return"] = np.log(c / c.shift(1))

    df["high_low_pct"]   = (h - l) / (c + eps)
    df["close_open_pct"] = (c - o) / (o + eps)

    # Price vs key MAs
    df["price_vs_ema50"]  = (c - df["ema_50"]) / (df["ema_50"] + eps)
    df["price_vs_ema200"] = (c - df["ema_200"]) / (df["ema_200"] + eps)

    # EMA crosses
    df["ema_cross_9_21"]  = df["ema_9"] - df["ema_21"]
    df["ema_cross_21_50"] = df["ema_21"] - df["ema_50"]

    # Volatility regime
    df["volatility_20"]    = df["log_return"].rolling(20).std() * np.sqrt(252 * 24)
    df["volatility_ratio"] = df["volatility_20"] / (df["volatility_20"].rolling(100).mean() + eps)

    # ═══ TIME & H1 TREND FEATURES (PRIORITY) ════════════════
    # Hour of day (cyclical) and day of week
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df["day_of_week"] = df.index.dayofweek
    
    # H1 Trend Proxy (80 EMA on M15 roughly equals 20 EMA on H1)
    df["ema_80"] = ta.trend.ema_indicator(c, window=80)
    df["h1_trend_slope"] = (df["ema_80"] - df["ema_80"].shift(4)) / (df["ema_80"].shift(4) + eps)

    # ═══ CANDLESTICK PATTERNS ═══════════════════════════════
    body = abs(c - o)
    candle_range = h - l + eps

    df["doji"]           = (body / candle_range < 0.1).astype(int)
    df["hammer"]         = ((c > o) & ((o - l) > 2 * body) & ((h - c) < body * 0.3)).astype(int)
    df["bullish_engulf"]  = (
        (c > o) &                          # Current is bullish
        (c.shift(1) < o.shift(1)) &        # Previous was bearish
        (c > o.shift(1)) &                 # Current close > prev open
        (o < c.shift(1))                   # Current open < prev close
    ).astype(int)
    df["bearish_engulf"]  = (
        (c < o) &
        (c.shift(1) > o.shift(1)) &
        (c < o.shift(1)) &
        (o > c.shift(1))
    ).astype(int)
    df["bullish_candle"] = (c > o).astype(int)
    df["shadow_ratio"]   = (h - c.combine(o, max)) / candle_range

    # ═══ RSI DIVERGENCE ═════════════════════════════════════
    # TODO: Improve divergence logic using proper swing high/low detection
    price_lower = (c < c.shift(5)) & (c < c.shift(10))
    rsi_higher  = (df["rsi_14"] > df["rsi_14"].shift(5))
    df["rsi_bull_div"] = (price_lower & rsi_higher).astype(int)

    price_higher = (c > c.shift(5)) & (c > c.shift(10))
    rsi_lower    = (df["rsi_14"] < df["rsi_14"].shift(5))
    df["rsi_bear_div"] = (price_higher & rsi_lower).astype(int)

    # ═══ LAG FEATURES (for time-series models) ══════════════
    for lag in [1, 2, 3, 5]:
        df[f"rsi_lag_{lag}"]       = df["rsi_14"].shift(lag)
        df[f"macd_hist_lag_{lag}"] = df["macd_hist"].shift(lag)
        df[f"returns_lag_{lag}"]   = df["returns_1"].shift(lag)

    # ═══ TARGET LABELS ══════════════════════════════════════
    # 1 if price goes up in next 12 bars, 0 if down
    df["target"] = (c.shift(-12) > c).astype(int)
    df["future_return"] = c.shift(-12) / c - 1

    # Explicitly drop rows where target is NaN (the last 12 rows) IMMEDIATELY
    # to prevent downstream code from accidentally training on future data.
    # Note: df.dropna(subset=['target']) behaves fine, but target is type int above, 
    # except astype(int) creates 0s instead of NaN for the condition `c.shift > c`.
    # It must be calculated correctly with NaN preservation:
    target_series = c.shift(-12) > c
    target_series.loc[c.shift(-12).isna()] = np.nan
    df["target"] = target_series

    df.dropna(subset=["target"], inplace=True)
    df["target"] = df["target"].astype(int)

    # Drop NaN rows (from rolling indicators)
    initial_len = len(df)
    df.dropna(inplace=True)
    logger.info(f"Features: {len(df.columns)} cols | {len(df)} rows ({initial_len - len(df)} dropped)")

    return df


def get_feature_columns() -> list:
    """Return the list of feature column names used for ML training."""
    return [
        # Trend
        "ema_9", "ema_21", "ema_50", "sma_20", "ema_200",
        "macd", "macd_signal", "macd_hist",
        "adx", "adx_pos", "adx_neg",
        "ichi_a", "ichi_b", "ichi_base", "ichi_conv",
        "h1_trend_slope",
        # Momentum
        "rsi_14", "rsi_7", "stoch_k", "stoch_d",
        "cci", "williams_r", "roc_10", "mfi_14",
        # Volatility
        "bb_width", "bb_pct", "atr_pct",
        "kc_upper", "kc_lower",
        # Volume
        "obv", "cmf_20", "vol_ratio",
        # Price-derived
        "returns_1", "returns_3", "returns_7", "returns_14",
        "log_return", "high_low_pct", "close_open_pct",
        "price_vs_ema50", "price_vs_ema200",
        "ema_cross_9_21", "ema_cross_21_50",
        "volatility_20", "volatility_ratio",
        # Time
        "hour_sin", "hour_cos", "day_of_week",
        # Candlestick patterns
        "doji", "hammer", "bullish_engulf", "bearish_engulf",
        "bullish_candle", "shadow_ratio",
        "rsi_bull_div", "rsi_bear_div",
        # Lags
        "rsi_lag_1", "rsi_lag_2", "rsi_lag_3", "rsi_lag_5",
        "macd_hist_lag_1", "macd_hist_lag_2", "macd_hist_lag_3", "macd_hist_lag_5",
        "returns_lag_1", "returns_lag_2", "returns_lag_3", "returns_lag_5",
    ]


def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Rolling z-score normalization.
    Each value normalized against its past 200-bar window.
    This prevents look-ahead bias (no future data leaks).
    """
    df_norm = df.copy()
    eps = np.finfo(float).eps
    
    binary_cols = {
        "doji", "hammer", "bullish_engulf", "bearish_engulf", 
        "bullish_candle", "rsi_bull_div", "rsi_bear_div",
        "day_of_week"
    }

    for col in feature_cols:
        if col in binary_cols or col.startswith("hour_"):
            # Do not normalize binary or cyclical time columns
            pass
        else:
            roll_mean = df[col].rolling(200, min_periods=50).mean()
            roll_std  = df[col].rolling(200, min_periods=50).std()
            df_norm[col] = (df[col] - roll_mean) / (roll_std + eps)

    # Safety net: fill any remaining NaN values after normalization
    nan_count = df_norm[feature_cols].isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values after normalization — forward-filling then zero-filling")
        df_norm[feature_cols] = df_norm[feature_cols].ffill().fillna(0)

    df_norm.dropna(inplace=True)
    return df_norm
