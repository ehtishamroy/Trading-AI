"""
Data Fetcher — Downloads and stores historical OHLCV data.
Pulls from MT5 (primary) with automated saving to Parquet files.
"""

import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DATA_DIR, MARKETS, ENTRY_TIMEFRAME, TREND_TIMEFRAME
from data.mt5_connector import connect_mt5, disconnect_mt5, get_ohlcv


def validate_ohlcv(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> pd.DataFrame:
    """
    Validate OHLCV data quality after fetching.
    Logs warnings for issues and cleans where possible.
    """
    if df.empty:
        return df

    label = f"{symbol} {timeframe}" if symbol else "data"
    issues = []

    # Check for null values
    null_counts = df[["open", "high", "low", "close", "volume"]].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        issues.append(f"{total_nulls} null values in OHLCV columns")
        df = df.dropna(subset=["open", "high", "low", "close"])

    # Check for duplicates
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate timestamps")
        df = df[~df.index.duplicated(keep="last")]

    # Check for time gaps (missing bars)
    if len(df) > 1:
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median()
        # A gap is >2x the median interval
        gaps = diffs[diffs > median_diff * 2]
        if len(gaps) > 0:
            issues.append(f"{len(gaps)} time gaps detected (>{median_diff * 2})")

    # Check for price anomalies (high < low)
    bad_bars = df[df["high"] < df["low"]]
    if len(bad_bars) > 0:
        issues.append(f"{len(bad_bars)} bars with high < low")

    if issues:
        for issue in issues:
            logger.warning(f"Data quality [{label}]: {issue}")
    else:
        logger.info(f"Data validation [{label}]: all checks passed")

    return df


def fetch_and_save(
    symbol: str,
    timeframe: str,
    num_bars: int = 50000
) -> pd.DataFrame:
    """
    Fetch OHLCV data from MT5 and save as Parquet.
    Parquet files are fast to load and small on disk.
    """
    df = get_ohlcv(symbol, timeframe, num_bars)
    if df.empty:
        logger.error(f"No data fetched for {symbol} {timeframe}")
        return df

    # Validate data quality
    df = validate_ohlcv(df, symbol, timeframe)

    # Save to parquet
    filename = f"{symbol}_{timeframe}.parquet"
    filepath = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath)
    logger.success(f"Saved {len(df)} bars → {filepath}")
    return df


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load saved data from Parquet file."""
    filepath = DATA_DIR / f"{symbol}_{timeframe}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(
            f"No data for {symbol} {timeframe}. "
            f"Run fetch_all_data() first."
        )
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df)} bars | {symbol} {timeframe}")
    return df


def fetch_all_data():
    """
    Download data for ALL configured markets + timeframes.
    Call this once to populate your data folder.

    Downloads:
    - EUR/USD: M15 + H1
    - XAU/USD: M15 + H1
    - BTC/USD: M15 + H1
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING ALL HISTORICAL DATA FROM MT5")
    logger.info("=" * 60)

    if not connect_mt5():
        logger.error("Cannot connect to MT5. Make sure MT5 is running!")
        return

    for market_key, market_info in MARKETS.items():
        symbol = market_info["mt5_symbol"]
        for tf in [ENTRY_TIMEFRAME, TREND_TIMEFRAME]:
            try:
                logger.info(f"\nFetching {symbol} {tf}...")
                fetch_and_save(symbol, tf, num_bars=50000)
            except Exception as e:
                logger.error(f"Failed {symbol} {tf}: {e}")

    disconnect_mt5()
    logger.success("\n✅ All data downloaded successfully!")
    logger.info(f"Data stored in: {DATA_DIR}")


def update_data(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    """
    Update existing data with latest bars.
    Used in live trading loop to keep data fresh.
    """
    filepath = DATA_DIR / f"{symbol}_{timeframe}.parquet"

    # Fetch latest bars
    new_df = get_ohlcv(symbol, timeframe, bars)
    if new_df.empty:
        return pd.DataFrame()

    # Merge with existing if available
    if filepath.exists():
        old_df = pd.read_parquet(filepath)
        combined = pd.concat([old_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_parquet(filepath)
        logger.info(f"Updated {symbol} {timeframe}: {len(combined)} total bars")
        return combined
    else:
        new_df.to_parquet(filepath)
        return new_df


if __name__ == "__main__":
    # Run standalone: python data/fetcher.py
    # This downloads ALL data for all markets
    fetch_all_data()
