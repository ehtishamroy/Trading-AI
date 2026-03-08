"""
Dead Man's Switch — Monitors system health and alerts if the bot stops.
Sends heartbeats every 5 minutes. If heartbeat is missed, sends EMERGENCY alert.
"""

import time
import threading
import asyncio
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


class DeadManSwitch:
    """
    Safety monitor that ensures the trading system is alive.

    If the system stops sending heartbeats:
    1. Sends EMERGENCY Telegram alert
    2. Logs the failure time
    3. Optionally closes all open positions

    This prevents a scenario where the bot crashes while
    positions are open and nobody notices.
    """

    def __init__(self, heartbeat_interval: int = 300, max_missed: int = 3):
        """
        Args:
            heartbeat_interval: Seconds between heartbeats (default 5 min)
            max_missed: Number of missed heartbeats before emergency (default 3)
        """
        self.heartbeat_interval = heartbeat_interval
        self.max_missed = max_missed
        self.last_heartbeat = datetime.now()
        self.is_alive = True
        self.monitor_thread = None

    def heartbeat(self):
        """Call this regularly from the main trading loop."""
        self.last_heartbeat = datetime.now()
        self.is_alive = True

    def start_monitoring(self):
        """Start the monitoring thread."""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Dead Man's Switch active — heartbeat every {self.heartbeat_interval}s")

    def _monitor_loop(self):
        """Background loop that checks for missed heartbeats."""
        missed = 0

        while True:
            time.sleep(self.heartbeat_interval)

            elapsed = (datetime.now() - self.last_heartbeat).total_seconds()

            if elapsed > self.heartbeat_interval * 1.5:
                missed += 1
                logger.warning(f"⚠️ Heartbeat missed ({missed}/{self.max_missed})")

                if missed >= self.max_missed:
                    self._trigger_emergency()
                    missed = 0  # Reset after emergency
            else:
                if missed > 0:
                    logger.info(f"Heartbeat recovered after {missed} missed beats")
                missed = 0

    def _trigger_emergency(self):
        """Send emergency alert — system appears to be dead."""
        self.is_alive = False
        logger.error("🚨 EMERGENCY — Trading system appears DEAD!")

        message = (
            "🚨🚨🚨 EMERGENCY ALERT 🚨🚨🚨\n\n"
            "Trading system has STOPPED responding!\n\n"
            f"Last heartbeat: {self.last_heartbeat.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Missed beats: {self.max_missed}\n\n"
            "ACTION REQUIRED:\n"
            "1. Check if MT5 is still running\n"
            "2. Check if Python process crashed\n"
            "3. Check open positions manually\n"
            "4. Restart: python main.py"
        )

        # Send via Telegram
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                import requests
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                requests.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                }, timeout=10)
            except Exception as e:
                logger.error(f"Emergency Telegram failed: {e}")

        # Log to file
        emergency_log = Path("logs/emergency.log")
        emergency_log.parent.mkdir(parents=True, exist_ok=True)
        with open(emergency_log, "a") as f:
            f.write(f"\n[{datetime.now().isoformat()}] SYSTEM DOWN - Last heartbeat: {self.last_heartbeat}\n")

    def get_status(self) -> dict:
        """Get current monitoring status."""
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return {
            "is_alive": self.is_alive,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "seconds_since": round(elapsed),
            "healthy": elapsed < self.heartbeat_interval * 1.5,
        }
