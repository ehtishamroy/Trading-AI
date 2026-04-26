"""
Feature Engineering — Transforms raw OHLCV into ML-ready features.
Includes technical indicators, microstructure features, session context,
book knowledge patterns, and triple-barrier target labels.
Uses rolling normalization to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
import ta
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    TRIPLE_BARRIER_TP_ATR, TRIPLE_BARRIER_SL_ATR, TRIPLE_BARRIER_MAX_HOLDING
)


# ═══ TRIPLE BARRIER LABELING ════════════════════════════════════════════

def _triple_barrier_labels(df: pd.DataFrame,
                           tp_atr_mult: float = TRIPLE_BARRIER_TP_ATR,
                           sl_atr_mult: float = TRIPLE_BARRIER_SL_ATR,
                           max_holding: int = TRIPLE_BARRIER_MAX_HOLDING) -> pd.Series:
    """
    Triple Barrier Method:
    For each bar, set TP = tp_atr_mult * ATR and SL = sl_atr_mult * ATR.
    Scan forward up to max_holding bars.
    Label 1 if TP hit first, 0 if SL hit first or timeout (price went down).
    Returns a Series of 0/1 labels aligned with df index.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr_14"].values
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(n - max_holding):
        entry = close[i]
        barrier_tp = entry + tp_atr_mult * atr[i]
        barrier_sl = entry - sl_atr_mult * atr[i]

        for j in range(1, max_holding + 1):
            idx = i + j
            if idx >= n:
                break
            # Check TP hit (high breaches upper barrier)
            if high[idx] >= barrier_tp:
                labels[i] = 1
                break
            # Check SL hit (low breaches lower barrier)
            if low[idx] <= barrier_sl:
                labels[i] = 0
                break
        else:
            # Timeout — use direction at expiry
            labels[i] = 1 if close[i + max_holding] > entry else 0

    return pd.Series(labels, index=df.index)


# ═══ MAIN FEATURE PIPELINE ════════════════════════════════════════════

def compute_all_features(df: pd.DataFrame, market_type: str = "forex") -> pd.DataFrame:
    """
    Compute ALL features from raw OHLCV data.
    Input: DataFrame with columns [open, high, low, close, volume]
    Output: Enriched DataFrame with ~45 quality features + triple-barrier target.
    """
    df = df.copy()
    eps = np.finfo(float).eps
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # ═══ TREND INDICATORS (kept: ema_21, ema_50, ema_200 — removed ema_9, sma_20) ═══
    df["ema_21"]  = ta.trend.ema_indicator(c, window=21)
    df["ema_50"]  = ta.trend.ema_indicator(c, window=50)
    df["ema_200"] = ta.trend.ema_indicator(c, window=200)

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

    # Ichimoku — collapsed to single signal: +1 above cloud, -1 below, 0 inside
    ichi = ta.trend.IchimokuIndicator(h, l)
    ichi_a = ichi.ichimoku_a()
    ichi_b = ichi.ichimoku_b()
    cloud_top = pd.concat([ichi_a, ichi_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([ichi_a, ichi_b], axis=1).min(axis=1)
    df["ichi_signal"] = np.where(c > cloud_top, 1, np.where(c < cloud_bot, -1, 0))

    # ═══ MOMENTUM INDICATORS (removed williams_r — redundant with stoch) ═══
    df["rsi_14"]   = ta.momentum.rsi(c, window=14)

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"]  = stoch.stoch()
    df["stoch_d"]  = stoch.stoch_signal()

    df["cci"]       = ta.trend.cci(h, l, c, window=20)
    df["roc_10"]    = ta.momentum.roc(c, window=10)
    df["mfi_14"]    = ta.volume.money_flow_index(h, l, c, v, window=14)

    # ═══ VOLATILITY INDICATORS (removed kc_upper/kc_lower) ════════════════
    bb = ta.volatility.BollingerBands(c)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()

    df["atr_14"]   = ta.volatility.average_true_range(h, l, c, window=14)
    df["atr_pct"]  = df["atr_14"] / c

    # ═══ NEW: VOLATILITY REGIME FEATURES ══════════════════════════════
    # ATR percentile rank over last 100 bars
    df["atr_percentile"] = df["atr_14"].rolling(100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    # BB squeeze duration: consecutive bars where BB width is below its 50-bar mean
    bb_mean = df["bb_width"].rolling(50, min_periods=10).mean()
    is_squeeze = (df["bb_width"] < bb_mean).astype(int)
    # Count consecutive squeeze bars
    squeeze_groups = is_squeeze.ne(is_squeeze.shift()).cumsum()
    df["bb_squeeze_duration"] = is_squeeze.groupby(squeeze_groups).cumsum()
    # Range contraction/expansion ratio
    range_20 = (h - l).rolling(20).mean()
    range_5 = (h - l).rolling(5).mean()
    df["range_ratio"] = range_5 / (range_20 + eps)

    # ═══ VOLUME INDICATORS (gated per market type) ══════════════════
    if market_type in ["crypto", "stock"]:
        df["obv"]        = ta.volume.on_balance_volume(c, v)
        df["cmf_20"]     = ta.volume.chaikin_money_flow(h, l, c, v, window=20)
        vol_sma_20 = v.rolling(20).mean()
        df["vol_ratio"]  = v / (vol_sma_20 + eps)
    else:
        df["obv"]        = 0.0
        df["cmf_20"]     = 0.0
        df["vol_ratio"]  = 0.0

    # ═══ PRICE-DERIVED FEATURES ═════════════════════════════════════
    df["returns_1"]  = c.pct_change(1)
    df["returns_3"]  = c.pct_change(3)
    df["returns_7"]  = c.pct_change(7)
    df["log_return"] = np.log(c / c.shift(1))

    df["high_low_pct"]   = (h - l) / (c + eps)
    df["close_open_pct"] = (c - o) / (o + eps)

    # Price vs key MAs (relative features — these are what matters, not raw EMAs)
    df["price_vs_ema50"]  = (c - df["ema_50"]) / (df["ema_50"] + eps)
    df["price_vs_ema200"] = (c - df["ema_200"]) / (df["ema_200"] + eps)

    # EMA crosses (relative)
    df["ema_cross_21_50"] = (df["ema_21"] - df["ema_50"]) / (df["ema_50"] + eps)

    # Volatility regime
    df["volatility_20"]    = df["log_return"].rolling(20).std() * np.sqrt(252 * 24)
    df["volatility_ratio"] = df["volatility_20"] / (df["volatility_20"].rolling(100).mean() + eps)

    # ═══ NEW: MICROSTRUCTURE FEATURES ═════════════════════════════════
    # Bar-to-bar velocity (acceleration of returns)
    df["return_accel"] = df["returns_1"] - df["returns_1"].shift(1)
    # Body-to-range ratio (conviction of the bar)
    body = abs(c - o)
    candle_range = h - l + eps
    df["body_ratio"] = body / candle_range
    # Price rejection wick ratio (long wicks = institutional orders)
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    df["upper_wick_ratio"] = upper_wick / candle_range
    df["lower_wick_ratio"] = lower_wick / candle_range

    # ═══ NEW: PERCENTILE RANK FEATURES ═══════════════════════════════
    df["rsi_percentile"] = df["rsi_14"].rolling(200, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df["macd_hist_percentile"] = df["macd_hist"].rolling(200, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ═══ TIME & H1 TREND FEATURES ═════════════════════════════════════
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df["day_of_week"] = df.index.dayofweek

    # H1 Trend Proxy (80 EMA on M15 roughly equals 20 EMA on H1)
    ema_80 = ta.trend.ema_indicator(c, window=80)
    df["h1_trend_slope"] = (ema_80 - ema_80.shift(4)) / (ema_80.shift(4) + eps)

    # ═══ CANDLESTICK PATTERNS ═══════════════════════════════════════
    df["doji"]           = (body / candle_range < 0.1).astype(int)
    df["hammer"]         = ((c > o) & ((o - l) > 2 * body) & ((h - c) < body * 0.3)).astype(int)
    df["bullish_engulf"]  = (
        (c > o) &
        (c.shift(1) < o.shift(1)) &
        (c > o.shift(1)) &
        (o < c.shift(1))
    ).astype(int)
    df["bearish_engulf"]  = (
        (c < o) &
        (c.shift(1) > o.shift(1)) &
        (c < o.shift(1)) &
        (o > c.shift(1))
    ).astype(int)
    df["bullish_candle"] = (c > o).astype(int)

    # ═══ RSI DIVERGENCE ═════════════════════════════════════════
    price_lower = (c < c.shift(5)) & (c < c.shift(10))
    rsi_higher  = (df["rsi_14"] > df["rsi_14"].shift(5))
    df["rsi_bull_div"] = (price_lower & rsi_higher).astype(int)

    price_higher = (c > c.shift(5)) & (c > c.shift(10))
    rsi_lower    = (df["rsi_14"] < df["rsi_14"].shift(5))
    df["rsi_bear_div"] = (price_higher & rsi_lower).astype(int)

    # ═══ NEW: INTERACTION FEATURES ═══════════════════════════════════
    # Volume spike AND EMA crossover (only meaningful for crypto/stock)
    vol_spike = (df["vol_ratio"] > 1.5).astype(int) if market_type in ["crypto", "stock"] else 0
    ema_cross_pos = (df["ema_cross_21_50"] > 0).astype(int)
    df["vol_spike_x_ema_cross"] = vol_spike * ema_cross_pos
    # RSI extreme AND trend aligned
    rsi_oversold = (df["rsi_14"] < 30).astype(int)
    rsi_overbought = (df["rsi_14"] > 70).astype(int)
    trend_up = (df["price_vs_ema50"] > 0).astype(int)
    df["rsi_extreme_x_trend"] = rsi_oversold * trend_up - rsi_overbought * (1 - trend_up)

    # ═══ SESSION FEATURES (from session_features.py) ════════════════
    try:
        from data.session_features import compute_session_features
        df = compute_session_features(df)
    except Exception as e:
        logger.warning(f"Session features failed: {e} — continuing without them")

    # ═══ BOOK KNOWLEDGE FEATURES (from book_knowledge.py) ═══════════
    try:
        from models.book_knowledge import BookKnowledge
        bk = BookKnowledge()
        df = bk.compute_book_features(df)
    except Exception as e:
        logger.warning(f"Book knowledge features failed: {e} — continuing without them")

    # ═══ TRIPLE BARRIER TARGET ════════════════════════════════════════
    df["target"] = _triple_barrier_labels(df)

    # Drop rows where target is NaN (last max_holding rows + any ATR warmup NaN)
    df.dropna(subset=["target"], inplace=True)
    df["target"] = df["target"].astype(int)

    # Drop NaN rows (from rolling indicators)
    initial_len = len(df)
    df.dropna(inplace=True)
    logger.info(f"Features: {len(df.columns)} cols | {len(df)} rows ({initial_len - len(df)} dropped)")

    return df


def get_feature_columns(market_type: str = "forex") -> list:
    """
    Return the list of feature column names used for ML training.
    ~45 quality features, market-type aware.
    """
    base_features = [
        # Trend (relative features only — no raw EMA values)
        "macd", "macd_signal", "macd_hist",
        "adx", "adx_pos", "adx_neg",
        "ichi_signal",
        "h1_trend_slope",
        "price_vs_ema50", "price_vs_ema200",
        "ema_cross_21_50",
        # Momentum
        "rsi_14", "stoch_k", "stoch_d",
        "cci", "roc_10", "mfi_14",
        # Volatility
        "bb_width", "bb_pct", "atr_pct",
        "atr_percentile", "bb_squeeze_duration", "range_ratio",
        # Volume (gated downstream — will be 0 for forex)
        "obv", "cmf_20", "vol_ratio",
        # Price-derived
        "returns_1", "returns_3", "returns_7",
        "log_return", "high_low_pct", "close_open_pct",
        "volatility_20", "volatility_ratio",
        # Microstructure
        "return_accel", "body_ratio",
        "upper_wick_ratio", "lower_wick_ratio",
        # Percentile ranks
        "rsi_percentile", "macd_hist_percentile",
        # Time
        "hour_sin", "hour_cos", "day_of_week",
        # Candlestick patterns
        "doji", "hammer", "bullish_engulf", "bearish_engulf",
        "bullish_candle",
        "rsi_bull_div", "rsi_bear_div",
        # Interaction features
        "vol_spike_x_ema_cross", "rsi_extreme_x_trend",
    ]

    # Session features (always available)
    session_features = [
        "sess_asian", "sess_london", "sess_new_york", "sess_overlap",
        "day_sin", "day_cos",
        "is_monday", "is_friday", "is_midweek",
        "session_progress", "volume_vs_session_avg",
        "dist_from_open", "near_daily_high", "near_daily_low",
    ]

    # Book knowledge features (always available)
    book_features = [
        "book_morning_star", "book_evening_star",
        "book_three_white_soldiers", "book_three_black_crows",
        "book_spinning_top",
        "book_bullish_marubozu", "book_bearish_marubozu",
        "book_tweezer_bottom", "book_tweezer_top",
        "book_bullish_ob", "book_bearish_ob",
        "book_bullish_fvg", "book_bearish_fvg", "book_fvg_size",
        "book_bullish_bos", "book_bearish_bos",
        "book_bull_sweep", "book_bear_sweep",
        "book_at_demand", "book_at_supply",
        "book_hidden_bull_div", "book_hidden_bear_div",
    ]

    return base_features + session_features + book_features


def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Rolling z-score normalization.
    Each value normalized against its past 200-bar window.
    This prevents look-ahead bias (no future data leaks).
    """
    df_norm = df.copy()
    eps = np.finfo(float).eps

    # Columns that should NOT be normalized (binary, cyclical, ordinal)
    skip_normalize = {
        "doji", "hammer", "bullish_engulf", "bearish_engulf",
        "bullish_candle", "rsi_bull_div", "rsi_bear_div",
        "day_of_week", "ichi_signal",
        "vol_spike_x_ema_cross", "rsi_extreme_x_trend",
        # Session binary flags
        "sess_asian", "sess_london", "sess_new_york", "sess_overlap",
        "is_monday", "is_friday", "is_midweek",
        # Book knowledge binary patterns
        "book_morning_star", "book_evening_star",
        "book_three_white_soldiers", "book_three_black_crows",
        "book_spinning_top",
        "book_bullish_marubozu", "book_bearish_marubozu",
        "book_tweezer_bottom", "book_tweezer_top",
        "book_bullish_ob", "book_bearish_ob",
        "book_bullish_fvg", "book_bearish_fvg",
        "book_bullish_bos", "book_bearish_bos",
        "book_bull_sweep", "book_bear_sweep",
        "book_at_demand", "book_at_supply",
        "book_hidden_bull_div", "book_hidden_bear_div",
    }

    for col in feature_cols:
        if col in skip_normalize or col.startswith("hour_") or col.startswith("day_"):
            pass
        elif col in df_norm.columns:
            roll_mean = df[col].rolling(200, min_periods=50).mean()
            roll_std  = df[col].rolling(200, min_periods=50).std()
            df_norm[col] = (df[col] - roll_mean) / (roll_std + eps)

    # Safety net: fill any remaining NaN values after normalization
    available_cols = [c for c in feature_cols if c in df_norm.columns]
    nan_count = df_norm[available_cols].isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values after normalization — forward-filling then zero-filling")
        df_norm[available_cols] = df_norm[available_cols].ffill().fillna(0)

    df_norm.dropna(inplace=True)
    return df_norm
