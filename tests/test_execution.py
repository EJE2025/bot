import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import execution
from trading_bot.exchanges import MockExchange


def test_open_and_close_position_success(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    monkeypatch.setattr(execution.time, "sleep", lambda x: None)

    order = execution.open_position("BTC_USDT", "BUY", 1, 30000, order_type="market")
    assert order["side"] == "buy"

    close = execution.close_position("BTC_USDT", "close_long", 1, order_type="market")
    assert close["side"] == "sell"


def test_open_position_retry_failure(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    monkeypatch.setattr(execution.time, "sleep", lambda x: None)

    def always_fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ex, "create_order", always_fail)

    with pytest.raises(execution.OrderSubmitError):
        execution.open_position("BTC_USDT", "BUY", 1, 30000, order_type="market")


def test_close_position_retry_success(monkeypatch):
    ex = MockExchange()
    monkeypatch.setattr(execution, "exchange", ex)
    monkeypatch.setattr(execution.time, "sleep", lambda x: None)

    original_create = ex.create_order
    calls = {"n": 0}

    def fail_once(*args, **kwargs):
        if calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("temp")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(ex, "create_order", fail_once)

    execution.open_position("BTC_USDT", "BUY", 1, 30000, order_type="market")
    order = execution.close_position("BTC_USDT", "close_long", 1, order_type="market")
    assert order["side"] == "sell"

