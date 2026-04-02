"""
Narrative Arc Tracker — Tracks the dominant market story over 7 days.
Claude uses this to understand the "plot" of the market, not just indicators.

Example narratives:
  - "EUR/USD has been falling for 3 days on hawkish Fed rhetoric"
  - "Gold broke out of a 2-week range on inflation fears"
  - "Bitcoin rejected $70k twice, bulls losing steam"
"""

import json
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from config.settings import MEMORY_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS


class NarrativeTracker:
    """
    Maintains a rolling 7-day market narrative.
    Updated daily — Claude reads it to understand the bigger picture.
    """

    def __init__(self, market: str = "EURUSD"):
        self.market = market
        self.narrative_file = MEMORY_DIR / f"narrative_{market}.json"
        self.narratives = self._load()

    def update_narrative(self, daily_summary: dict, news_context: str = ""):
        """
        Add today's market story to the rolling narrative.

        Args:
            daily_summary: {
                date, open, close, high, low, change_pct,
                regime, key_levels, trades_taken, pnl
            }
            news_context: Today's top headlines
        """
        entry = {
            "date": daily_summary.get("date", datetime.now().strftime("%Y-%m-%d")),
            "price_action": self._summarize_price_action(daily_summary),
            "regime": daily_summary.get("regime", "UNKNOWN"),
            "key_events": news_context[:500] if news_context else "No major news",
            "trades": daily_summary.get("trades_taken", 0),
            "pnl": daily_summary.get("pnl", 0),
            "key_levels": daily_summary.get("key_levels", {}),
        }

        self.narratives.append(entry)

        # Keep only last 7 days
        self.narratives = self.narratives[-7:]
        self._save()

        logger.info(f"Narrative updated for {self.market} — {len(self.narratives)} days tracked")

    def get_narrative_for_claude(self) -> str:
        """
        Generate the current market narrative for Claude's context.
        This gives Claude a multi-day view of what's happening.
        """
        if not self.narratives:
            return f"## Market Narrative ({self.market})\nNo narrative history yet. System just started — first day of trading."

        lines = [
            f"## Market Narrative ({self.market}) — Last {len(self.narratives)} Days",
            f"_This is the running story of {self.market}. Use it to understand the bigger picture._\n",
        ]

        for entry in self.narratives:
            date = entry.get("date", "?")
            action = entry.get("price_action", "")
            regime = entry.get("regime", "?")
            events = entry.get("key_events", "")
            pnl = entry.get("pnl", 0)

            pnl_emoji = "📈" if pnl > 0 else ("📉" if pnl < 0 else "➖")
            lines.append(f"**{date}** ({regime})")
            lines.append(f"  {action}")
            if events and events != "No major news":
                lines.append(f"  📰 {events[:150]}")
            if entry.get("trades", 0) > 0:
                lines.append(f"  {pnl_emoji} {entry['trades']} trades, ${pnl:+.2f} PnL")
            lines.append("")

        # Add trend summary
        if len(self.narratives) >= 3:
            changes = [n.get("price_action", "") for n in self.narratives[-3:]]
            lines.append("**Recent Trend**: " + " → ".join(changes[-3:]))

        return "\n".join(lines)

    def generate_ai_narrative(self, market_data: str) -> str:
        """
        Use local Ollama LLM to generate a narrative summary of recent market action.
        Called at end of each trading day. FREE — no API key needed.
        """
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a market analyst. Write a brief 2-3 sentence narrative about today's market action. Focus on the story — what happened, why, and what it means for tomorrow. Be concise."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize today's action for {self.market}:\n{market_data}"
                    }
                ],
                "stream": False,
                "options": {"num_predict": 500, "temperature": 0.7}
            }
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama narrative error: {e}")
            return ""

    def _summarize_price_action(self, summary: dict) -> str:
        """Create a brief price action summary."""
        change = summary.get("change_pct", 0)
        high = summary.get("high", 0)
        low = summary.get("low", 0)
        close = summary.get("close", 0)

        if change > 0.5:
            action = f"Strong rally +{change:.2f}%"
        elif change > 0.1:
            action = f"Mild upside +{change:.2f}%"
        elif change > -0.1:
            action = f"Flat / consolidated"
        elif change > -0.5:
            action = f"Mild decline {change:.2f}%"
        else:
            action = f"Sharp selloff {change:.2f}%"

        return f"{action} (H:{high} L:{low} C:{close})"

    def _load(self) -> list:
        if self.narrative_file.exists():
            with open(self.narrative_file) as f:
                return json.load(f)
        return []

    def _save(self):
        self.narrative_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.narrative_file, "w") as f:
            json.dump(self.narratives, f, indent=2, default=str)
