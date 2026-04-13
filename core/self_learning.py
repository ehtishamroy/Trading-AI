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
        Record a trade entry with all context.

        trade should contain:
        - market, direction, entry_price, ticket, lot_size
        - aegis_score, sl, tp, status
        """
        entry = {
            "id": len(self.journal) + 1,
            "timestamp": datetime.now().isoformat(),
            "ticket": trade.get("ticket", 0),
            "market": trade.get("market", ""),
            "direction": trade.get("direction", ""),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price", 0),
            "pnl": trade.get("pnl", 0),
            "pnl_pct": trade.get("pnl_pct", 0),
            "result": "OPEN",
            "status": trade.get("status", "OPEN"),
            "aegis_score": trade.get("aegis_score", 0),
            "sl": trade.get("sl", 0),
            "tp": trade.get("tp", 0),
            "lot_size": trade.get("lot_size", 0.01),
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
        logger.info(f"Trade #{entry['id']} recorded: {entry['direction']} {entry['market']} (ticket: {entry['ticket']})")

    def reconcile_closed_trades(self):
        """
        Pull closed positions from MT5 history and update journal entries
        with exit_price, PnL, and status=CLOSED. Then update pattern memory.
        """
        try:
            from data.mt5_connector import get_trade_history
        except ImportError:
            logger.warning("Cannot import get_trade_history — skipping reconciliation")
            return 0

        closed_deals = get_trade_history(days=7)
        if not closed_deals:
            return 0

        updated = 0
        for deal in closed_deals:
            # Find matching open journal entry by ticket/position_id
            for entry in self.journal:
                if entry.get("status") != "OPEN":
                    continue
                # Match by ticket number
                entry_ticket = entry.get("ticket", 0)
                if entry_ticket and (entry_ticket == deal.get("position_id") or entry_ticket == deal.get("ticket")):
                    pnl = deal["profit"] + deal.get("commission", 0) + deal.get("swap", 0)
                    entry["exit_price"] = deal["price"]
                    entry["pnl"] = round(pnl, 2)
                    entry["pnl_pct"] = round(pnl / max(entry.get("entry_price", 1) * entry.get("lot_size", 0.01), 0.01) * 100, 2)
                    entry["result"] = "WIN" if pnl > 0 else "LOSS"
                    entry["status"] = "CLOSED"
                    entry["close_time"] = deal.get("time", datetime.now()).isoformat() if isinstance(deal.get("time"), datetime) else str(deal.get("time", ""))
                    updated += 1
                    logger.info(f"Trade #{entry['id']} reconciled: {entry['result']} (${pnl:.2f})")
                    # Update pattern memory with actual outcome
                    self._update_patterns(entry)
                    break

        if updated > 0:
            self._save_json(self.journal_file, self.journal)
            logger.info(f"Reconciled {updated} closed trade(s) from MT5 history")

        return updated

    def _update_patterns(self, trade: dict):
        """Update the pattern memory with this trade's outcome.

        NOTE: This records EVERY outcome unconditionally — no sample-size gating
        happens here. Statistical filtering (the 10-trade minimum) lives on the
        READ side in get_pattern_stats() and get_pattern_memory_for_claude().
        Do NOT add a minimum here: recording must continue during the learning
        phase, otherwise the 10-trade threshold would be unreachable forever.
        """
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

        IMPORTANT: Patterns with fewer than MIN_PATTERN_SAMPLES trades are
        statistically meaningless and are EXCLUDED from the per-row display.
        Exposing raw win rates on 3–5 sample sizes causes Claude to rationalize
        HOLD on noise, producing a negative feedback loop where the system
        cannot build new history. Mirrors the 10-trade threshold used by
        get_pattern_stats().
        """
        MIN_PATTERN_SAMPLES = 10

        lines = ["## Pattern Memory (Historical Performance)"]
        relevant = {k: v for k, v in self.patterns.items() if market in k}

        if not relevant:
            lines.append(
                "No historical patterns recorded yet. System is in initial learning phase — "
                "pattern memory is NEUTRAL and should NOT bias your decision. "
                "Rely on ML ensemble, regime, and price action."
            )
            return "\n".join(lines)

        # Split by statistical significance
        significant = {k: v for k, v in relevant.items() if v["total"] >= MIN_PATTERN_SAMPLES}
        learning = {k: v for k, v in relevant.items() if v["total"] < MIN_PATTERN_SAMPLES}

        if significant:
            lines.append("### Statistically Significant Patterns (>= 10 trades)")
            for key, data in sorted(significant.items(), key=lambda x: x[1]["total"], reverse=True):
                emoji = "✅" if data["win_rate"] > 0.55 else ("⚠️" if data["win_rate"] > 0.40 else "❌")
                lines.append(
                    f"- {emoji} **{key}**: {data['win_rate']:.0%} win rate "
                    f"({data['total']} trades, ${data['total_pnl']:.2f} total PnL)"
                )

        if learning:
            total_learning = sum(v["total"] for v in learning.values())
            lines.append("")
            lines.append("### Learning Phase (< 10 trades — NOT statistically significant)")
            lines.append(
                f"- {len(learning)} pattern(s) with {total_learning} total trade(s) — "
                f"sample size too small to be meaningful. These are EXCLUDED from "
                f"decision logic. Treat pattern memory as NEUTRAL (0.5) and do NOT "
                f"cite 'historical win rate' as a reason to HOLD. System needs more "
                f"trades to learn — taking reasonable setups now builds the dataset."
            )

        return "\n".join(lines)

    def get_pattern_stats(self, market: str, regime: str, direction: str) -> float:
        """
        Get historical win rate for a specific pattern.
        Returns win rate (0.0-1.0) or 0.5 if no data exists.
        """
        key = f"{market}_{regime}_{direction}"
        if key in self.patterns and self.patterns[key]["total"] >= 10:
            return self.patterns[key]["win_rate"]
        # Fall back to market-level stats if specific pattern has too few trades
        market_patterns = {k: v for k, v in self.patterns.items() if k.startswith(market)}
        if market_patterns:
            total_wins = sum(v["wins"] for v in market_patterns.values())
            total_trades = sum(v["total"] for v in market_patterns.values())
            if total_trades >= 10:
                return round(total_wins / total_trades, 3)
        return 0.5  # Default when not enough historical data (need 10+ trades)

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
