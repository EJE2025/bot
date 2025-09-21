import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import metrics
from trading_bot import notify


def test_maybe_alert_respects_cooldown(monkeypatch):
    sent = []

    def fake_send(message):
        sent.append(message)

    monkeypatch.setattr(notify, "send_telegram", fake_send)
    monkeypatch.setattr(notify, "send_discord", fake_send)
    metrics.maybe_alert(True, "alert", cooldown=1)
    metrics.maybe_alert(True, "alert", cooldown=1)
    assert sent.count("alert") == 2  # both telegram and discord once
