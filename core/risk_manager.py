"""
Risk Management Engine — Protects capital above all else.
Handles position sizing, stop losses, circuit breakers.
"""

import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    MAX_RISK_PER_TRADE, MAX_TRADES_PER_DAY, MAX_CONSECUTIVE_LOSSES,
    PAUSE_AFTER_LOSSES_HOURS, MAX_DAILY_DRAWDOWN, MAX_TOTAL_DRAWDOWN,
    MIN_REWARD_RISK, ATR_STOP_MULTIPLIER, LOT_SIZE, STARTING_CAPITAL,
)


class RiskManager:
    """
    Risk management engine. Hard-coded safety rules
    that CANNOT be overridden by ML models or Claude.
    """

    def __init__(self):
        self.trades_today = 0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.starting_balance = STARTING_CAPITAL
        self.paused_until = None

    def can_trade(self, balance: float) -> dict:
        """
        Check ALL safety rules before allowing a trade.
        Returns {allowed: bool, reason: str}
        """
        from datetime import datetime

        # Rule 1: Check pause from consecutive losses
        if self.paused_until and datetime.now() < self.paused_until:
            remaining = (self.paused_until - datetime.now()).seconds // 60
            return {"allowed": False, "reason": f"Paused for {remaining} more minutes (consecutive losses)"}

        # Rule 2: Max trades per day
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return {"allowed": False, "reason": f"Max {MAX_TRADES_PER_DAY} trades/day reached"}

        # Rule 3: Daily drawdown
        if balance > 0:
            daily_dd = self.daily_pnl / balance
            if daily_dd < -MAX_DAILY_DRAWDOWN:
                return {"allowed": False, "reason": f"Daily drawdown limit hit ({daily_dd:.1%})"}

        # Rule 4: Total drawdown
        total_dd = (balance - self.starting_balance) / self.starting_balance
        if total_dd < -MAX_TOTAL_DRAWDOWN:
            return {"allowed": False, "reason": f"TOTAL DRAWDOWN LIMIT HIT ({total_dd:.1%}). System shutting off."}

        return {"allowed": True, "reason": "All checks passed ✅"}

    def calculate_position(self, balance: float, atr: float, price: float,
                           regime_multiplier: float = 1.0) -> dict:
        """
        Calculate position size, stop loss, and take profit.

        Args:
            balance: Current account balance
            atr: Current ATR (Average True Range) value
            price: Current entry price
            regime_multiplier: From regime detector (0.5 to 1.0)

        Returns:
            {lot_size, stop_loss_pips, take_profit_pips, risk_amount}
        """
        # Risk amount = 2% of balance, adjusted by regime
        risk_amount = balance * MAX_RISK_PER_TRADE * regime_multiplier

        # Stop loss = ATR-based
        stop_loss_distance = atr * ATR_STOP_MULTIPLIER

        # Take profit = minimum 2:1 reward-to-risk
        take_profit_distance = stop_loss_distance * MIN_REWARD_RISK

        # Position size = fixed micro lot (0.01) for $200 account
        lot_size = LOT_SIZE

        return {
            "lot_size": lot_size,
            "risk_amount": round(risk_amount, 2),
            "stop_loss_distance": round(stop_loss_distance, 5),
            "take_profit_distance": round(take_profit_distance, 5),
            "reward_risk_ratio": MIN_REWARD_RISK,
            "regime_multiplier": regime_multiplier,
        }

    def calculate_sl_tp(self, entry_price: float, direction: str,
                        sl_distance: float, tp_distance: float) -> dict:
        """Calculate exact SL and TP prices."""
        if direction.lower() == "buy":
            sl = round(entry_price - sl_distance, 5)
            tp = round(entry_price + tp_distance, 5)
        else:
            sl = round(entry_price + sl_distance, 5)
            tp = round(entry_price - tp_distance, 5)

        return {
            "entry": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "direction": direction,
        }

    def record_trade(self, pnl: float):
        """Record a completed trade for tracking."""
        self.trades_today += 1
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                from datetime import datetime, timedelta
                self.paused_until = datetime.now() + timedelta(hours=PAUSE_AFTER_LOSSES_HOURS)
                logger.warning(
                    f"⚠️ {self.consecutive_losses} consecutive losses! "
                    f"Trading paused until {self.paused_until.strftime('%H:%M')}"
                )
        else:
            self.consecutive_losses = 0  # Reset on a win

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.paused_until = None
        logger.info("Daily risk counters reset")

    def get_status(self) -> dict:
        """Get current risk status."""
        return {
            "trades_today": self.trades_today,
            "max_trades": MAX_TRADES_PER_DAY,
            "consecutive_losses": self.consecutive_losses,
            "daily_pnl": round(self.daily_pnl, 2),
            "paused": self.paused_until is not None,
        }
