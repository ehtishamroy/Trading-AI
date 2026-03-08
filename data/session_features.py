"""
Session & Time Features — Adds trading session context.
London-NY overlap, Asian session, time-of-day patterns.
Markets behave differently at different times of day.
"""

import pandas as pd
import numpy as np
from loguru import logger


# ─── Session Definitions (UTC) ───────────────────────────
SESSIONS = {
    "asian":  {"start": 0,  "end": 8},    # 00:00-08:00 UTC (Tokyo/Sydney)
    "london": {"start": 7,  "end": 16},   # 07:00-16:00 UTC
    "new_york": {"start": 12, "end": 21}, # 12:00-21:00 UTC
    "overlap": {"start": 12, "end": 16},  # London-NY overlap (highest volume)
}


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session/time-based features to OHLCV DataFrame.
    These capture the cyclical patterns of when markets are active.
    """
    df = df.copy()

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    hour = df.index.hour
    day = df.index.dayofweek  # 0=Mon, 4=Fri

    # ═══ SESSION FLAGS ══════════════════════════════════════
    df["sess_asian"]    = ((hour >= 0) & (hour < 8)).astype(int)
    df["sess_london"]   = ((hour >= 7) & (hour < 16)).astype(int)
    df["sess_new_york"] = ((hour >= 12) & (hour < 21)).astype(int)
    df["sess_overlap"]  = ((hour >= 12) & (hour < 16)).astype(int)  # Best time to trade

    # ═══ TIME ENCODINGS (cyclical) ══════════════════════════
    # Encode hour as sin/cos to capture cyclical nature (23→0 is close)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Encode day of week (Mon→Fri)
    df["day_sin"] = np.sin(2 * np.pi * day / 5)
    df["day_cos"] = np.cos(2 * np.pi * day / 5)

    # ═══ DAY-OF-WEEK FLAGS ══════════════════════════════════
    df["is_monday"]  = (day == 0).astype(int)  # Often ranging / gap fills
    df["is_friday"]  = (day == 4).astype(int)  # Position squaring
    df["is_midweek"] = ((day >= 1) & (day <= 3)).astype(int)  # Tue-Thu = best moves

    # ═══ SESSION-RELATIVE FEATURES ══════════════════════════
    # How far into the session are we? (0.0 = start, 1.0 = end)
    df["session_progress"] = _calculate_session_progress(hour)

    # ═══ VOLUME-BY-SESSION ══════════════════════════════════
    # Average volume for this hour (rolling 20-day average for same hour)
    df["hour_of_day"] = hour
    df["avg_volume_for_hour"] = df.groupby("hour_of_day")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    df["volume_vs_session_avg"] = df["volume"] / (df["avg_volume_for_hour"] + 1e-9)

    # ═══ OPENING RANGE ══════════════════════════════════════
    # First 1-hour range of the trading session (key support/resistance)
    df["daily_open"] = df.groupby(df.index.date)["open"].transform("first")
    df["dist_from_open"] = (df["close"] - df["daily_open"]) / (df["daily_open"] + 1e-9)

    # ═══ HIGH-OF-DAY / LOW-OF-DAY PROXIMITY ═════════════════
    df["daily_high"] = df.groupby(df.index.date)["high"].transform("cummax")
    df["daily_low"] = df.groupby(df.index.date)["low"].transform(lambda x: x.expanding().min())
    daily_range = df["daily_high"] - df["daily_low"] + 1e-9
    df["near_daily_high"] = (df["daily_high"] - df["close"]) / daily_range
    df["near_daily_low"]  = (df["close"] - df["daily_low"]) / daily_range

    # ═══ DAY-OF-WEEK VOLATILITY RATIO ═══════════════════════
    # Is today more/less volatile than average for this day of week?
    df["day_of_week"] = day
    df["avg_range_for_day"] = df.groupby("day_of_week")["high_low_pct"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    ) if "high_low_pct" in df.columns else 0

    # Cleanup temp columns
    df.drop(columns=["hour_of_day", "day_of_week"], errors="ignore", inplace=True)

    logger.info(f"Session features added: {len([c for c in df.columns if c.startswith('sess_')])} session cols")
    return df


def _calculate_session_progress(hour: pd.Series) -> pd.Series:
    """Calculate how far into the active trading session we are."""
    progress = pd.Series(0.5, index=hour.index)

    # During London-NY overlap (most active)
    overlap = (hour >= 12) & (hour < 16)
    progress[overlap] = (hour[overlap] - 12) / 4.0

    # During London only
    london_only = (hour >= 7) & (hour < 12)
    progress[london_only] = (hour[london_only] - 7) / 9.0

    # During NY only
    ny_only = (hour >= 16) & (hour < 21)
    progress[ny_only] = (hour[ny_only] - 12) / 9.0

    return progress


def get_session_feature_columns() -> list:
    """Return column names of session features for ML training."""
    return [
        "sess_asian", "sess_london", "sess_new_york", "sess_overlap",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "is_monday", "is_friday", "is_midweek",
        "session_progress", "volume_vs_session_avg",
        "dist_from_open", "near_daily_high", "near_daily_low",
    ]


def get_best_trading_hours(market: str) -> dict:
    """Get optimal trading hours for each market."""
    optimal = {
        "EURUSD": {
            "best": "12:00-16:00 UTC (London-NY Overlap)",
            "good": "07:00-12:00 UTC (London), 16:00-19:00 UTC (NY)",
            "avoid": "22:00-05:00 UTC (Asian — low EUR volume)",
            "pkt": "5:00 PM - 9:00 PM PKT (Overlap)",
        },
        "XAUUSD": {
            "best": "12:00-18:00 UTC (NY Open + Overlap)",
            "good": "07:00-12:00 UTC (London)",
            "avoid": "22:00-05:00 UTC (Low volume)",
            "pkt": "5:00 PM - 11:00 PM PKT (NY Session)",
        },
        "BTCUSD": {
            "best": "13:00-20:00 UTC (US trading hours)",
            "good": "07:00-13:00 UTC (EU trading hours)",
            "avoid": "Weekends have lower liquidity",
            "pkt": "6:00 PM - 1:00 AM PKT (US hours)",
        },
    }
    return optimal.get(market, optimal["EURUSD"])
