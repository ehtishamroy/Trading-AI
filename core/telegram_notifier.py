"""
Telegram Notifier
Sends alerts directly to your phone when a high-confidence trade is found.
"""

import requests
import sys
from loguru import logger
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PROXY_URL

def send_telegram_alert(message: str) -> bool:
    """Send a text message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram config missing, skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        # HTML is much safer for AI-generated text than Telegram's strict Markdown
        "parse_mode": "HTML"
    }

    # SOCKS5 Proxy to bypass regional blocks (loaded from PROXY_URL env var)
    proxies = {}
    if PROXY_URL:
        proxies = {"http": PROXY_URL, "https": PROXY_URL}
    
    try:
        response = requests.post(url, json=payload, proxies=proxies, timeout=15)
        response.raise_for_status()
        logger.info("📲 Telegram alert sent successfully.")
        return True
    except requests.exceptions.HTTPError as e:
        # If Telegram rejects the HTML formatting, strip it and send as raw text
        if response.status_code == 400:
            logger.warning("Telegram rejected HTML formatting. Retrying as raw text...")
            payload.pop("parse_mode", None)
            try:
                raw_response = requests.post(url, json=payload, proxies=proxies, timeout=15)
                raw_response.raise_for_status()
                logger.info("📲 Telegram alert sent successfully (Raw Text string).")
                return True
            except Exception as e2:
                logger.error(f"Failed to send raw Telegram alert: {e2}")
        else:
            logger.error(f"Failed to send Telegram alert: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False
