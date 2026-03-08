"""
Telegram Notifier
Sends alerts directly to your phone when a high-confidence trade is found.
"""

import requests
import sys
from loguru import logger
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_alert(message: str) -> bool:
    """Send a text message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram config missing, skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    # SOCKS5 Proxy to bypass regional blocks
    # Using socks5h:// to tunnel DNS requests as well (crucial for bypassing regional blocks)
    proxies = {
        "http": "socks5h://1dc54f765e72f72cac00__cr.gb:c3a6db9859e7c9cc@gw.dataimpulse.com:10003",
        "https": "socks5h://1dc54f765e72f72cac00__cr.gb:c3a6db9859e7c9cc@gw.dataimpulse.com:10003"
    }
    
    try:
        response = requests.post(url, json=payload, proxies=proxies, timeout=15)
        response.raise_for_status()
        logger.info("📲 Telegram alert sent successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False
