"""
Telegram Alert Bot — Sends trade signals to your phone.
Shows Claude's reasoning, Aegis Score, and approve/reject buttons.
"""

import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from loguru import logger
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


class TradingBot:
    """
    Telegram bot for trade alerts and human approval.

    Features:
    - Sends trade signals with Claude's reasoning
    - Approve/Reject buttons on each signal
    - Daily summary reports
    - Status commands (/status, /stats, /pause, /resume)
    """

    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.app = None
        self.pending_trades = {}  # Store trades waiting for approval
        self.auto_mode = False

        if not self.token:
            logger.warning("No TELEGRAM_BOT_TOKEN — Telegram alerts disabled")

    async def send_trade_alert(self, analysis: dict):
        """
        Send a trade alert with approve/reject buttons.
        """
        if not self.token or not self.chat_id:
            return

        verdict = analysis.get("verdict", {})
        aegis = analysis.get("aegis", {})
        market = analysis.get("market", "UNKNOWN")
        debate = analysis.get("debate", {})

        decision = verdict.get("decision", "HOLD")
        confidence = verdict.get("confidence", 0)
        entry = verdict.get("entry_price", "N/A")
        sl = verdict.get("stop_loss", "N/A")
        tp = verdict.get("take_profit", "N/A")
        reasoning = verdict.get("reasoning", "No reasoning provided")
        pre_mortem = verdict.get("pre_mortem", "")
        risks = verdict.get("key_risks", [])

        # Build Aegis bar
        score = aegis.get("score", 0)
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        level_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(aegis.get("level", "RED"), "⚪")

        # Emoji for decision
        dec_emoji = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸️"}.get(decision, "❓")

        message = (
            f"{dec_emoji} **{decision} {market}**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⭐ **Aegis Score**: {score}/100 {level_emoji}\n"
            f"[{bar}]\n\n"
            f"📊 **Entry**: {entry}\n"
            f"🛑 **Stop Loss**: {sl}\n"
            f"🎯 **Take Profit**: {tp}\n"
            f"🎲 **Confidence**: {confidence}/10\n\n"
            f"💭 **Why**: {reasoning[:500]}\n\n"
        )

        if pre_mortem:
            message += f"⚠️ **Pre-mortem**: {pre_mortem[:200]}\n\n"

        if risks:
            message += "🚨 **Risks**:\n"
            for r in risks[:3]:
                message += f"  • {r}\n"
            message += "\n"

        # Bull vs Bear summary
        bull_case = debate.get("bull_case", "")[:200]
        bear_case = debate.get("bear_case", "")[:200]
        if bull_case and bear_case:
            message += (
                f"🟢 **Bull**: {bull_case}...\n\n"
                f"🔴 **Bear**: {bear_case}...\n"
            )

        # Store trade for approval
        trade_id = f"trade_{len(self.pending_trades) + 1}"
        self.pending_trades[trade_id] = analysis

        # Auto-mode: approve automatically
        if self.auto_mode and aegis.get("level") == "GREEN":
            message += "\n✅ **AUTO-APPROVED** (Auto-mode ON + Green Aegis)\n"
            await self._send_message(message)
            return {"approved": True, "auto": True}

        # Manual mode: add approve/reject buttons
        keyboard = [
            [
                InlineKeyboardButton("✅ APPROVE", callback_data=f"approve_{trade_id}"),
                InlineKeyboardButton("❌ REJECT", callback_data=f"reject_{trade_id}"),
            ],
            [
                InlineKeyboardButton("✏️ Modify SL/TP", callback_data=f"modify_{trade_id}"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await self._send_message(message, reply_markup=reply_markup)
        return {"approved": False, "pending": trade_id}

    async def send_daily_summary(self, stats: dict):
        """Send end-of-day trading summary."""
        if not self.token or not self.chat_id:
            return

        pnl_emoji = "📈" if stats.get("pnl", 0) >= 0 else "📉"
        message = (
            f"📊 **Daily Summary**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔢 Trades: {stats.get('trades', 0)}\n"
            f"✅ Wins: {stats.get('wins', 0)}\n"
            f"❌ Losses: {stats.get('losses', 0)}\n"
            f"{pnl_emoji} PnL: ${stats.get('pnl', 0):.2f}\n"
        )
        await self._send_message(message)

    async def send_alert(self, message: str):
        """Send a plain text alert."""
        if self.token and self.chat_id:
            await self._send_message(f"🔔 {message}")

    async def _send_message(self, text: str, reply_markup=None):
        """Send a message to the configured chat."""
        try:
            from telegram import Bot
            bot = Bot(token=self.token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    # ─── Command Handlers (for interactive bot) ──────────

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        await update.message.reply_text(
            "🟢 Trading System Online\n"
            f"Mode: {'AUTO' if self.auto_mode else 'MANUAL'}\n"
            f"Pending trades: {len(self.pending_trades)}"
        )

    async def cmd_auto_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /auto command — enable auto-trading."""
        self.auto_mode = True
        await update.message.reply_text("🤖 Auto-mode ENABLED. Green signals will auto-execute.")

    async def cmd_auto_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /manual command — disable auto-trading."""
        self.auto_mode = False
        await update.message.reply_text("👤 Manual mode ENABLED. You approve every trade.")

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle approve/reject button presses."""
        query = update.callback_query
        await query.answer()
        data = query.data

        if data.startswith("approve_"):
            trade_id = data.replace("approve_", "")
            await query.edit_message_text(f"✅ Trade APPROVED — executing...\n(ID: {trade_id})")
            # TODO: Trigger actual trade execution
            logger.info(f"Trade {trade_id} approved by user")

        elif data.startswith("reject_"):
            trade_id = data.replace("reject_", "")
            await query.edit_message_text(f"❌ Trade REJECTED\n(ID: {trade_id})")
            logger.info(f"Trade {trade_id} rejected by user")

        elif data.startswith("modify_"):
            trade_id = data.replace("modify_", "")
            await query.edit_message_text(
                f"✏️ Send new SL and TP:\n"
                f"Format: /modify {trade_id} SL=1.0850 TP=1.0920"
            )

    def run_bot(self):
        """Start the Telegram bot (blocking — run in separate thread)."""
        if not self.token:
            logger.warning("Cannot start Telegram bot — no token")
            return

        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("auto", self.cmd_auto_on))
        self.app.add_handler(CommandHandler("manual", self.cmd_auto_off))
        self.app.add_handler(CallbackQueryHandler(self.handle_button))

        logger.info("🤖 Telegram bot started")
        self.app.run_polling()
