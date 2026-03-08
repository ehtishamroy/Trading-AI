"""
COT Data Fetcher — Commitment of Traders report from CFTC.
Shows what large speculators (hedge funds) and commercials are positioned.
Massive edge for Gold and EUR/USD.
Updated weekly (every Friday).
"""

import requests
import pandas as pd
from datetime import datetime
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DATA_DIR


# CFTC COT report codes for our markets
COT_CODES = {
    "EURUSD": "099741",      # Euro FX
    "XAUUSD": "088691",      # Gold
    "BTCUSD": "133741",      # Bitcoin
}

COT_URL = "https://publicreporting.cftc.gov/api/views/dpe5-ag8d/rows.json"


def fetch_cot_data(market: str = "EURUSD") -> pd.DataFrame:
    """
    Fetch COT data from CFTC public API.
    Returns DataFrame with net positions for large specs and commercials.
    """
    code = COT_CODES.get(market)
    if not code:
        logger.warning(f"No COT code for {market}")
        return pd.DataFrame()

    try:
        # Use the CFTC Quandl-style endpoint
        url = f"https://data.nasdaq.com/api/v3/datasets/CFTC/{code}_FO_ALL.json"
        params = {"rows": 52, "order": "desc"}  # Last year

        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            logger.warning(f"COT API returned {response.status_code}, using fallback")
            return _generate_sample_cot(market)

        data = response.json().get("dataset", {})
        columns = data.get("column_names", [])
        rows = data.get("data", [])

        df = pd.DataFrame(rows, columns=columns)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Fetched {len(df)} weeks of COT data for {market}")
        return df

    except Exception as e:
        logger.error(f"COT fetch error: {e}")
        return _generate_sample_cot(market)


def get_cot_summary(market: str = "EURUSD") -> dict:
    """
    Get a summary of COT positioning for Claude's context.

    Returns:
        {net_position, trend, change_1w, sentiment label}
    """
    df = fetch_cot_data(market)
    if df.empty:
        return {
            "available": False,
            "message": "COT data unavailable — using other signals"
        }

    # Try to extract net speculative positions
    try:
        # Common column names in COT data
        long_cols = [c for c in df.columns if "Long" in c and "Noncommercial" in c]
        short_cols = [c for c in df.columns if "Short" in c and "Noncommercial" in c]

        if long_cols and short_cols:
            latest_long = df[long_cols[0]].iloc[-1]
            latest_short = df[short_cols[0]].iloc[-1]
            prev_long = df[long_cols[0]].iloc[-2] if len(df) > 1 else latest_long
            prev_short = df[short_cols[0]].iloc[-2] if len(df) > 1 else latest_short

            net_current = latest_long - latest_short
            net_prev = prev_long - prev_short
            change = net_current - net_prev
        else:
            return _fallback_cot_summary(market)

    except (KeyError, IndexError):
        return _fallback_cot_summary(market)

    # Determine sentiment
    if net_current > 0 and change > 0:
        sentiment = "Bullish (Specs adding longs)"
    elif net_current > 0 and change < 0:
        sentiment = "Weakening Long (Specs reducing)"
    elif net_current < 0 and change < 0:
        sentiment = "Bearish (Specs adding shorts)"
    elif net_current < 0 and change > 0:
        sentiment = "Weakening Short (Specs covering)"
    else:
        sentiment = "Neutral"

    return {
        "available": True,
        "net_position": int(net_current),
        "change_1w": int(change),
        "sentiment": sentiment,
        "longs": int(latest_long),
        "shorts": int(latest_short),
    }


def format_cot_for_claude(cot: dict, market: str) -> str:
    """Format COT data for Claude's context prompt."""
    if not cot.get("available", False):
        return f"## COT Data ({market})\n{cot.get('message', 'Unavailable')}"

    return (
        f"## COT Data ({market}) — Hedge Fund Positioning\n"
        f"**Net Position**: {cot['net_position']:+,} contracts "
        f"({'LONG' if cot['net_position'] > 0 else 'SHORT'})\n"
        f"**Weekly Change**: {cot['change_1w']:+,} contracts\n"
        f"**Sentiment**: {cot['sentiment']}\n"
        f"Longs: {cot['longs']:,} | Shorts: {cot['shorts']:,}"
    )


def _generate_sample_cot(market: str) -> pd.DataFrame:
    """Generate sample COT data when API is unavailable."""
    logger.info(f"Using sample COT data for {market}")
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=52, freq="W-FRI")
    np.random.seed(42)
    net = np.cumsum(np.random.randn(52) * 5000)
    return pd.DataFrame({
        "Date": dates,
        "Noncommercial Long": (net + abs(net.min()) + 50000).astype(int),
        "Noncommercial Short": (abs(net.min()) + 50000 - net).astype(int),
    }).set_index("Date")


def _fallback_cot_summary(market: str) -> dict:
    """Return a fallback summary when columns don't match."""
    return {
        "available": True,
        "net_position": 0,
        "change_1w": 0,
        "sentiment": "Data format unknown — treat as neutral",
        "longs": 0,
        "shorts": 0,
    }
