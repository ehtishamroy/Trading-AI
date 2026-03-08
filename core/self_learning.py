"""
Self-Learning Engine — The system's memory and growth mechanism.
Records every trade, learns from wins/losses, builds pattern memory.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import JOURNAL_DIR, MEMORY_DIR


class TradingJournal:
    """
    Maintains a complete log of every trade + Claude's reasoning.
    Used for daily reviews, pattern tracking, and weekly retraining.
    """

    def __init__(self):
        self.journal_file = JOURNAL_DIR / "trade_journal.json"
        self.memory_file = MEMORY_DIR / "pattern_memory.json"
        self.journal = self._load_json(self.journal_file, default=[])
        self.patterns = self._load_json(self.memory_file, default={})

    def record_trade(self, trade: dict):
        """
        Record a completed trade with all context.

        trade should contain:
        - market, direction, entry_price, exit_price, pnl
        - aegis_score, ml_signals, claude_reasoning
        - regime, timestamp, model_version
        """
        entry = {
            "id": len(self.journal) + 1,
            "timestamp": datetime.now().isoformat(),
            "market": trade.get("market", ""),
            "direction": trade.get("direction", ""),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price", 0),
            "pnl": trade.get("pnl", 0),
            "pnl_pct": trade.get("pnl_pct", 0),
            "result": "WIN" if trade.get("pnl", 0) > 0 else "LOSS",
            "aegis_score": trade.get("aegis_score", 0),
            "ml_confidence": trade.get("ml_confidence", 0),
            "claude_confidence": trade.get("claude_confidence", 0),
            "claude_reasoning": trade.get("claude_reasoning", ""),
            "regime": trade.get("regime", ""),
            "signals_used": trade.get("signals_used", {}),
            "model_version": trade.get("model_version", "v1"),
            "debrief": trade.get("debrief", ""),
        }

        self.journal.append(entry)
        self._save_json(self.journal_file, self.journal)
        logger.info(f"Trade #{entry['id']} recorded: {entry['result']} (${entry['pnl']:.2f})")

        # Update pattern memory
        self._update_patterns(entry)

    def _update_patterns(self, trade: dict):
        """Update the pattern memory with this trade's outcome."""
        # Create a pattern key from the trade's characteristics
        regime = trade.get("regime", "UNKNOWN")
        direction = trade.get("direction", "unknown")
        market = trade.get("market", "UNKNOWN")

        key = f"{market}_{regime}_{direction}"

        if key not in self.patterns:
            self.patterns[key] = {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0,
                "avg_aegis": 0,
            }

        p = self.patterns[key]
        p["total"] += 1
        if trade["pnl"] > 0:
            p["wins"] += 1
        else:
            p["losses"] += 1
        p["total_pnl"] += trade["pnl"]
        p["win_rate"] = round(p["wins"] / p["total"], 3) if p["total"] > 0 else 0
        p["avg_aegis"] = round(
            (p["avg_aegis"] * (p["total"] - 1) + trade.get("aegis_score", 0)) / p["total"], 1
        )

        self._save_json(self.memory_file, self.patterns)

    def get_pattern_memory_for_claude(self, market: str, regime: str) -> str:
        """
        Format pattern memory as context for Claude.
        Claude uses this to make decisions based on past performance.
        """
        lines = ["## Pattern Memory (Historical Performance)"]
        relevant = {k: v for k, v in self.patterns.items() if market in k}

        if not relevant:
            lines.append("No historical patterns recorded yet. System is learning.")
            return "\n".join(lines)

        for key, data in sorted(relevant.items(), key=lambda x: x[1]["total"], reverse=True):
            emoji = "✅" if data["win_rate"] > 0.55 else ("⚠️" if data["win_rate"] > 0.40 else "❌")
            lines.append(
                f"- {emoji} **{key}**: {data['win_rate']:.0%} win rate "
                f"({data['total']} trades, ${data['total_pnl']:.2f} total PnL)"
            )

        return "\n".join(lines)

    def get_daily_summary(self) -> dict:
        """Get today's trading summary."""
        today = datetime.now().date().isoformat()
        today_trades = [t for t in self.journal if t["timestamp"].startswith(today)]

        if not today_trades:
            return {"trades": 0, "pnl": 0, "wins": 0, "losses": 0}

        return {
            "trades": len(today_trades),
            "pnl": round(sum(t["pnl"] for t in today_trades), 2),
            "wins": sum(1 for t in today_trades if t["pnl"] > 0),
            "losses": sum(1 for t in today_trades if t["pnl"] <= 0),
            "best_trade": max(today_trades, key=lambda t: t["pnl"]),
            "worst_trade": min(today_trades, key=lambda t: t["pnl"]),
        }

    def get_overall_stats(self) -> dict:
        """Get lifetime trading statistics."""
        if not self.journal:
            return {"total": 0}

        wins = [t for t in self.journal if t["pnl"] > 0]
        losses = [t for t in self.journal if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in self.journal)
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

        return {
            "total_trades": len(self.journal),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self.journal), 3),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            "best_trade": max(self.journal, key=lambda t: t["pnl"]),
            "worst_trade": min(self.journal, key=lambda t: t["pnl"]),
        }

    def get_confidence_calibration(self) -> dict:
        """
        Check if confidence scores match actual outcomes.
        E.g., if 80% confidence trades only win 50%, model is overconfident.
        """
        if len(self.journal) < 10:
            return {"status": "Not enough data (need 10+ trades)"}

        # Group trades by confidence buckets
        buckets = {"low": [], "mid": [], "high": []}
        for t in self.journal:
            aegis = t.get("aegis_score", 0)
            if aegis < 70:
                buckets["low"].append(t)
            elif aegis < 85:
                buckets["mid"].append(t)
            else:
                buckets["high"].append(t)

        calibration = {}
        for bucket, trades in buckets.items():
            if trades:
                expected_wr = {"low": 0.5, "mid": 0.6, "high": 0.75}[bucket]
                actual_wr = sum(1 for t in trades if t["pnl"] > 0) / len(trades)
                calibration[bucket] = {
                    "trades": len(trades),
                    "expected_win_rate": expected_wr,
                    "actual_win_rate": round(actual_wr, 3),
                    "calibrated": abs(actual_wr - expected_wr) < 0.15,
                }

        return calibration

    @staticmethod
    def _load_json(path: Path, default=None):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default if default is not None else {}

    @staticmethod
    def _save_json(path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
