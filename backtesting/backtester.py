"""
Backtesting Framework — Validates strategies on historical data.
Uses vectorbt for fast vectorized backtesting.

Rules:
- Walk-forward validation (no look-ahead bias)
- Time-series splits (never shuffle time data)
- Must pass minimum metrics before going live
"""

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    INITIAL_CAPITAL, COMMISSION_RATE,
    BACKTEST_START, BACKTEST_END, LOT_SIZE,
)


class Backtester:
    """
    Backtesting engine for the ML trading system.
    Tests models on unseen historical data to validate performance.
    """

    MINIMUM_REQUIREMENTS = {
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.15,    # Max 15% drawdown
        "win_rate": 0.40,         # Min 40% (with 2:1 R:R this is profitable)
        "profit_factor": 1.3,
        "total_trades": 30,       # Need min 30 trades for statistical significance
    }

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = [initial_capital]

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> dict:
        """
        Run a backtest on historical signals.

        Args:
            signals: DataFrame with columns [signal, confidence]
                     signal: 1 = buy, -1 = sell, 0 = hold
            prices: DataFrame with OHLCV columns
            stop_loss_pct: Stop loss as % of entry
            take_profit_pct: Take profit as % of entry

        Returns:
            dict with performance metrics
        """
        capital = self.initial_capital
        position = None
        self.trades = []
        self.equity_curve = [capital]

        for i in range(len(signals)):
            signal = signals.iloc[i].get("signal", 0)
            confidence = signals.iloc[i].get("confidence", 0.5)
            price = prices.iloc[i]["close"]
            high = prices.iloc[i]["high"]
            low = prices.iloc[i]["low"]

            # Check if in position — manage SL/TP
            if position:
                if position["direction"] == "long":
                    # Check stop loss
                    if low <= position["stop_loss"]:
                        pnl = (position["stop_loss"] - position["entry"]) / position["entry"]
                        pnl_usd = capital * abs(pnl) * (-1)
                        capital += pnl_usd
                        self.trades.append({
                            "entry": position["entry"],
                            "exit": position["stop_loss"],
                            "direction": "long",
                            "pnl_pct": pnl,
                            "pnl_usd": pnl_usd,
                            "result": "LOSS",
                            "bars_held": i - position["entry_bar"],
                        })
                        position = None

                    # Check take profit
                    elif high >= position["take_profit"]:
                        pnl = (position["take_profit"] - position["entry"]) / position["entry"]
                        pnl_usd = capital * abs(pnl)
                        capital += pnl_usd
                        self.trades.append({
                            "entry": position["entry"],
                            "exit": position["take_profit"],
                            "direction": "long",
                            "pnl_pct": pnl,
                            "pnl_usd": pnl_usd,
                            "result": "WIN",
                            "bars_held": i - position["entry_bar"],
                        })
                        position = None

                elif position["direction"] == "short":
                    if high >= position["stop_loss"]:
                        pnl = (position["entry"] - position["stop_loss"]) / position["entry"]
                        pnl_usd = capital * abs(pnl) * (-1)
                        capital += pnl_usd
                        self.trades.append({
                            "entry": position["entry"],
                            "exit": position["stop_loss"],
                            "direction": "short",
                            "pnl_pct": pnl,
                            "pnl_usd": pnl_usd,
                            "result": "LOSS",
                            "bars_held": i - position["entry_bar"],
                        })
                        position = None

                    elif low <= position["take_profit"]:
                        pnl = (position["entry"] - position["take_profit"]) / position["entry"]
                        pnl_usd = capital * abs(pnl)
                        capital += pnl_usd
                        self.trades.append({
                            "entry": position["entry"],
                            "exit": position["take_profit"],
                            "direction": "short",
                            "pnl_pct": pnl,
                            "pnl_usd": pnl_usd,
                            "result": "WIN",
                            "bars_held": i - position["entry_bar"],
                        })
                        position = None

            # Open new position if no position and signal exists
            if position is None and signal != 0 and confidence >= 0.6:
                if signal == 1:  # Buy
                    position = {
                        "direction": "long",
                        "entry": price,
                        "stop_loss": price * (1 - stop_loss_pct),
                        "take_profit": price * (1 + take_profit_pct),
                        "entry_bar": i,
                    }
                elif signal == -1:  # Sell
                    position = {
                        "direction": "short",
                        "entry": price,
                        "stop_loss": price * (1 + stop_loss_pct),
                        "take_profit": price * (1 - take_profit_pct),
                        "entry_bar": i,
                    }

            # Apply commission
            if self.trades and self.trades[-1].get("entry_bar", 0) == i:
                capital -= capital * COMMISSION_RATE

            self.equity_curve.append(capital)

        return self.calculate_metrics()

    def calculate_metrics(self) -> dict:
        """Calculate all performance metrics."""
        if not self.trades:
            return {"error": "No trades executed"}

        wins = [t for t in self.trades if t["result"] == "WIN"]
        losses = [t for t in self.trades if t["result"] == "LOSS"]

        total_pnl = sum(t["pnl_usd"] for t in self.trades)
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        avg_win = np.mean([t["pnl_usd"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl_usd"] for t in losses])) if losses else 1

        # Sharpe Ratio
        returns = pd.Series([t["pnl_pct"] for t in self.trades])
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)

        # Max Drawdown
        equity = pd.Series(self.equity_curve)
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Profit Factor
        gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        metrics = {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 3),
            "total_pnl_usd": round(total_pnl, 2),
            "total_return_pct": round((self.equity_curve[-1] / self.initial_capital - 1) * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd, 3),
            "avg_bars_held": round(np.mean([t["bars_held"] for t in self.trades]), 1),
            "final_equity": round(self.equity_curve[-1], 2),
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"BACKTEST RESULTS")
        logger.info(f"{'='*50}")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

        return metrics

    def passes_minimum(self, metrics: dict) -> dict:
        """
        Check if backtest results meet minimum requirements.
        Model CANNOT go live unless it passes these checks.
        """
        results = {}
        for key, threshold in self.MINIMUM_REQUIREMENTS.items():
            value = metrics.get(key, 0)
            if key == "max_drawdown":
                passed = value >= threshold  # DD is negative, so >= -0.15 is OK
            else:
                passed = value >= threshold
            results[key] = {
                "value": value,
                "threshold": threshold,
                "passed": passed,
            }

        all_passed = all(r["passed"] for r in results.values())
        results["overall"] = "PASSED ✅" if all_passed else "FAILED ❌"

        if all_passed:
            logger.success("Model PASSES minimum requirements — approved for demo trading")
        else:
            failed = [k for k, v in results.items() if k != "overall" and not v.get("passed")]
            logger.warning(f"Model FAILS: {', '.join(failed)} — NOT approved for trading")

        return results

    def walk_forward_test(
        self,
        df: pd.DataFrame,
        model_fn,
        n_splits: int = 5,
    ) -> list[dict]:
        """
        Walk-forward testing — the gold standard for ML trading validation.

        Splits data into rolling windows:
        [Train1 | Test1]
              [Train2 | Test2]
                    [Train3 | Test3]
                          [Train4 | Test4]
                                [Train5 | Test5]

        Each test window is unseen by the model that makes predictions on it.
        """
        results = []
        total_len = len(df)
        window_size = total_len // (n_splits + 1)

        for i in range(n_splits):
            train_start = i * window_size
            train_end = train_start + window_size * 2
            test_start = train_end
            test_end = min(test_start + window_size, total_len)

            if test_end > total_len:
                break

            df_train = df.iloc[train_start:train_end]
            df_test = df.iloc[test_start:test_end]

            logger.info(f"\nWalk-Forward Split {i+1}/{n_splits}")
            logger.info(f"  Train: {len(df_train)} bars | Test: {len(df_test)} bars")

            # Train model on this window
            model = model_fn(df_train)

            # Get predictions on test window
            predictions = model.predict(df_test)

            # Run backtest on this test window
            signals = pd.DataFrame({"signal": predictions, "confidence": 0.7})
            metrics = self.run_backtest(signals, df_test)
            metrics["split"] = i + 1
            results.append(metrics)

        # Aggregate results
        if results:
            avg_sharpe = np.mean([r.get("sharpe_ratio", 0) for r in results])
            avg_wr = np.mean([r.get("win_rate", 0) for r in results])
            avg_pf = np.mean([r.get("profit_factor", 0) for r in results])
            logger.info(f"\nWalk-Forward Summary:")
            logger.info(f"  Avg Sharpe: {avg_sharpe:.2f}")
            logger.info(f"  Avg Win Rate: {avg_wr:.2%}")
            logger.info(f"  Avg Profit Factor: {avg_pf:.2f}")

        return results
