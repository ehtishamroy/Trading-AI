"""
News Sentiment Pipeline — Fetches market news and scores sentiment.
Output feeds into Claude's reasoning and ML features.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import NEWS_API_KEY


# Keyword mapping for each market
MARKET_QUERIES = {
    "EURUSD": "EUR USD forex ECB Federal Reserve interest rates",
    "XAUUSD": "gold price XAU USD inflation safe haven",
    "BTCUSD": "bitcoin BTC crypto cryptocurrency",
}


def fetch_news(market: str = "EURUSD", hours_back: int = 12) -> list[dict]:
    """
    Fetch recent news headlines from NewsAPI.
    Returns list of {title, description, source, published_at}
    """
    if not NEWS_API_KEY:
        logger.warning("No NEWS_API_KEY — using empty news context")
        return []

    query = MARKET_QUERIES.get(market, "forex trading")
    from_time = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_time,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 15,
                "apiKey": NEWS_API_KEY,
            },
            timeout=10,
        )
        articles = resp.json().get("articles", [])
        logger.info(f"Fetched {len(articles)} articles for {market}")
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
            }
            for a in articles if a.get("title")
        ]
    except Exception as e:
        logger.error(f"News fetch error: {e}")
        return []


def score_sentiment(text: str) -> float:
    """
    Simple keyword-based sentiment: -1.0 (bearish) to +1.0 (bullish).
    Will be replaced by FinBERT once ML stack is running.
    """
    bullish = [
        "bullish", "rally", "surge", "breakout", "adoption",
        "etf approved", "all-time high", "buy", "gains",
        "growth", "approve", "partnership", "rate cut",
        "dovish", "stimulus", "easing", "recovery",
    ]
    bearish = [
        "bearish", "crash", "ban", "hack", "regulation", "lawsuit",
        "sell-off", "dump", "fraud", "bankrupt", "warning",
        "fear", "decline", "reject", "rate hike", "hawkish",
        "tightening", "recession", "inflation",
    ]

    text_lower = text.lower()
    bull = sum(1 for w in bullish if w in text_lower)
    bear = sum(1 for w in bearish if w in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


def get_sentiment_summary(market: str) -> dict:
    """
    Get complete sentiment summary for a market.
    Returns {score, label, headlines} for Claude context.
    """
    articles = fetch_news(market)
    if not articles:
        return {"score": 0.0, "label": "Neutral", "headline_count": 0, "top_headlines": []}

    scores = [score_sentiment(f"{a['title']} {a.get('description', '')}") for a in articles]
    avg = sum(scores) / len(scores)

    label = "Bullish" if avg > 0.15 else ("Bearish" if avg < -0.15 else "Neutral")

    return {
        "score": round(avg, 3),
        "label": label,
        "headline_count": len(articles),
        "top_headlines": [a["title"] for a in articles[:5]],
    }


def format_for_claude(sentiment: dict, market: str) -> str:
    """Format sentiment data as markdown for Claude's prompt."""
    lines = [f"## News Sentiment ({market}) — Last 12 Hours"]
    lines.append(f"**Overall**: {sentiment['label']} (score: {sentiment['score']})")
    lines.append(f"Based on {sentiment['headline_count']} articles.")
    if sentiment["top_headlines"]:
        lines.append("\n**Top Headlines:**")
        for h in sentiment["top_headlines"]:
            lines.append(f"  - {h}")
    return "\n".join(lines)
