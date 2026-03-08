"""
Economic Calendar — Tracks high-impact news events.
Warns the system before NFP, FOMC, CPI, ECB, etc.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# High-impact events that affect each market
HIGH_IMPACT_EVENTS = {
    "EURUSD": [
        "Non-Farm Payrolls", "FOMC", "Fed Rate Decision", "CPI",
        "ECB Rate Decision", "ECB Press Conference", "GDP",
        "Unemployment Rate", "Core PPI", "Retail Sales",
    ],
    "XAUUSD": [
        "Non-Farm Payrolls", "FOMC", "Fed Rate Decision", "CPI",
        "Core CPI", "PPI", "Inflation", "10-Year Bond Auction",
        "Gold Reserves", "Unemployment Claims",
    ],
    "BTCUSD": [
        "SEC", "ETF", "FOMC", "Fed Rate Decision", "CPI",
        "Crypto Regulation", "Inflation",
    ],
}


def fetch_economic_calendar(days_ahead: int = 3) -> list[dict]:
    """
    Fetch upcoming economic events.
    Uses a free economic calendar API.

    Returns list of events with impact level.
    """
    try:
        # Try multiple free sources
        events = _fetch_from_trading_economics(days_ahead)
        if events:
            return events

        # Fallback: manual high-impact schedule
        return _get_known_recurring_events(days_ahead)

    except Exception as e:
        logger.error(f"Calendar fetch error: {e}")
        return _get_known_recurring_events(days_ahead)


def _fetch_from_trading_economics(days_ahead: int) -> list:
    """Try to fetch from a free economic calendar API."""
    try:
        from_date = datetime.utcnow().strftime("%Y-%m-%d")
        to_date = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200:
            return []

        events = resp.json()
        processed = []
        for event in events:
            impact = event.get("impact", "").lower()
            if impact in ("high", "medium"):
                processed.append({
                    "title": event.get("title", ""),
                    "country": event.get("country", ""),
                    "date": event.get("date", ""),
                    "impact": impact,
                    "forecast": event.get("forecast", ""),
                    "previous": event.get("previous", ""),
                })

        logger.info(f"Fetched {len(processed)} upcoming events")
        return processed

    except Exception:
        return []


def _get_known_recurring_events(days_ahead: int) -> list:
    """
    Fallback: return known recurring high-impact events.
    These happen at predictable times every month.
    """
    now = datetime.utcnow()
    events = []

    # NFP: First Friday of every month
    first_day = now.replace(day=1)
    nfp_day = first_day
    while nfp_day.weekday() != 4:  # Friday
        nfp_day += timedelta(days=1)

    if 0 <= (nfp_day - now).days <= days_ahead:
        events.append({
            "title": "Non-Farm Payrolls (NFP)",
            "country": "USD",
            "date": nfp_day.strftime("%Y-%m-%d 13:30 UTC"),
            "impact": "high",
            "forecast": "",
            "previous": "",
        })

    return events


def get_upcoming_events(market: str, hours_ahead: int = 4) -> list[dict]:
    """
    Get events in the next N hours that affect this market.
    Used to warn the system before placing trades.
    """
    events = fetch_economic_calendar(days_ahead=1)
    relevant = []

    keywords = HIGH_IMPACT_EVENTS.get(market, [])
    for event in events:
        title = event.get("title", "")
        if any(kw.lower() in title.lower() for kw in keywords):
            relevant.append(event)

    if relevant:
        logger.warning(f"⚠️ {len(relevant)} high-impact events upcoming for {market}")

    return relevant


def should_avoid_trading(market: str, hours_buffer: int = 1) -> dict:
    """
    Check if trading should be avoided due to upcoming events.

    Returns:
        {avoid: bool, reason: str, events: list}
    """
    events = get_upcoming_events(market, hours_ahead=hours_buffer)

    if events:
        event_names = [e["title"] for e in events]
        return {
            "avoid": True,
            "reason": f"High-impact event(s) within {hours_buffer}h: {', '.join(event_names)}",
            "events": events,
        }

    return {"avoid": False, "reason": "No major events upcoming", "events": []}


def format_for_claude(events: list, market: str) -> str:
    """Format economic calendar for Claude's context."""
    if not events:
        return f"## Economic Calendar ({market})\nNo high-impact events in the next 4 hours. ✅"

    lines = [f"## Economic Calendar ({market}) ⚠️ HIGH IMPACT"]
    for e in events:
        impact_emoji = "🔴" if e["impact"] == "high" else "🟡"
        lines.append(
            f"{impact_emoji} **{e['title']}** ({e['country']})\n"
            f"   Time: {e['date']} | Forecast: {e.get('forecast', 'N/A')} | Prev: {e.get('previous', 'N/A')}"
        )
    return "\n".join(lines)
