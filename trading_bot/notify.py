import requests
from . import config


def send_telegram(message: str):
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception:
        pass


def send_discord(message: str):
    if not config.DISCORD_WEBHOOK:
        return
    try:
        requests.post(config.DISCORD_WEBHOOK, json={"content": message}, timeout=10)
    except Exception:
        pass

