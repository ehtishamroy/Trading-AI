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
