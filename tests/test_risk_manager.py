"""Tests for RiskManager — circuit breakers and position sizing."""

import pytest
from core.risk_manager import RiskManager


class TestCanTrade:
    def test_allows_first_trade(self):
        rm = RiskManager()
        result = rm.can_trade(500.0)
        assert result["allowed"] is True

    def test_blocks_after_max_trades(self):
        rm = RiskManager()
        rm.trades_today = 4
        result = rm.can_trade(500.0)
        assert result["allowed"] is False
        assert "Max" in result["reason"]

    def test_blocks_on_daily_drawdown(self):
        rm = RiskManager()
        rm.daily_pnl = -30.0  # -6% on $500
        result = rm.can_trade(500.0)
        assert result["allowed"] is False
        assert "drawdown" in result["reason"].lower()

    def test_blocks_on_total_drawdown(self):
        rm = RiskManager()
        rm.starting_balance = 500.0
        result = rm.can_trade(400.0)  # -20% from starting
        assert result["allowed"] is False
        assert "TOTAL" in result["reason"]

    def test_blocks_during_pause(self):
        from datetime import datetime, timedelta
        rm = RiskManager()
        rm.paused_until = datetime.now() + timedelta(hours=1)
        result = rm.can_trade(500.0)
        assert result["allowed"] is False
        assert "Paused" in result["reason"]

    def test_allows_after_pause_expires(self):
        from datetime import datetime, timedelta
        rm = RiskManager()
        rm.paused_until = datetime.now() - timedelta(hours=1)
        result = rm.can_trade(500.0)
        assert result["allowed"] is True


class TestRecordTrade:
    def test_win_resets_consecutive_losses(self):
        rm = RiskManager()
        rm.record_trade(-5.0)
        rm.record_trade(-5.0)
        assert rm.consecutive_losses == 2
        rm.record_trade(10.0)
        assert rm.consecutive_losses == 0

    def test_consecutive_losses_trigger_pause(self):
        rm = RiskManager()
        rm.record_trade(-5.0)
        rm.record_trade(-5.0)
        rm.record_trade(-5.0)
        assert rm.paused_until is not None

    def test_daily_pnl_tracking(self):
        rm = RiskManager()
        rm.record_trade(10.0)
        rm.record_trade(-3.0)
        assert rm.daily_pnl == 7.0
        assert rm.trades_today == 2


class TestResetDaily:
    def test_resets_counters(self):
        rm = RiskManager()
        rm.trades_today = 3
        rm.daily_pnl = -10.0
        rm.reset_daily()
        assert rm.trades_today == 0
        assert rm.daily_pnl == 0.0
        assert rm.paused_until is None


class TestPositionSizing:
    def test_basic_calculation(self):
        rm = RiskManager()
        pos = rm.calculate_position(500.0, 0.0015, 1.1000)
        assert pos["lot_size"] == 0.01
        assert pos["risk_amount"] > 0
        assert pos["stop_loss_distance"] > 0
        assert pos["take_profit_distance"] > pos["stop_loss_distance"]

    def test_regime_multiplier_reduces_risk(self):
        rm = RiskManager()
        normal = rm.calculate_position(500.0, 0.0015, 1.1000, regime_multiplier=1.0)
        reduced = rm.calculate_position(500.0, 0.0015, 1.1000, regime_multiplier=0.5)
        assert reduced["risk_amount"] < normal["risk_amount"]

    def test_sl_tp_buy(self):
        rm = RiskManager()
        prices = rm.calculate_sl_tp(1.1000, "buy", 0.0015, 0.0030)
        assert prices["stop_loss"] < prices["entry"]
        assert prices["take_profit"] > prices["entry"]

    def test_sl_tp_sell(self):
        rm = RiskManager()
        prices = rm.calculate_sl_tp(1.1000, "sell", 0.0015, 0.0030)
        assert prices["stop_loss"] > prices["entry"]
        assert prices["take_profit"] < prices["entry"]
