import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

from trading_bot import permissions, config
from trading_bot.exchanges import MockExchange


def test_can_open_trade_in_test_mode(monkeypatch):
    monkeypatch.setattr(config, "TEST_MODE", True, raising=False)
    monkeypatch.setattr(config, "TRADING_MODE", "paper", raising=False)
    monkeypatch.setattr(config, "ALLOW_LIVE_TRADING", False, raising=False)
    assert permissions.can_open_trade()


def test_live_trading_requires_flag(monkeypatch):
    monkeypatch.setattr(config, "TEST_MODE", False, raising=False)
    monkeypatch.setattr(config, "TRADING_MODE", "live", raising=False)
    monkeypatch.setattr(config, "ALLOW_LIVE_TRADING", False, raising=False)
    with pytest.raises(permissions.PermissionError):
        permissions.ensure_open_trade_allowed()


def test_mock_exchange_bypasses_live_checks(monkeypatch):
    monkeypatch.setattr(config, "TEST_MODE", False, raising=False)
    monkeypatch.setattr(config, "TRADING_MODE", "live", raising=False)
    monkeypatch.setattr(config, "ALLOW_LIVE_TRADING", False, raising=False)
    permissions.ensure_open_trade_allowed(MockExchange())


def test_missing_credentials_with_uppercase_default_exchange(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_EXCHANGE", "BINANCE", raising=False)
    monkeypatch.setattr(config, "BINANCE_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "BINANCE_API_SECRET", "", raising=False)

    missing = permissions._missing_credentials()

    assert missing == ["BINANCE_API_KEY", "BINANCE_API_SECRET"]


def test_token_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TEST_MODE", False, raising=False)
    monkeypatch.setattr(config, "TRADING_MODE", "live", raising=False)
    monkeypatch.setattr(config, "ALLOW_LIVE_TRADING", True, raising=False)
    monkeypatch.setattr(config, "DEFAULT_EXCHANGE", "bitget", raising=False)
    monkeypatch.setattr(config, "BITGET_API_KEY", "k", raising=False)
    monkeypatch.setattr(config, "BITGET_API_SECRET", "s", raising=False)
    monkeypatch.setattr(config, "BITGET_PASSPHRASE", "p", raising=False)

    token = tmp_path / "live-token.txt"
    token.write_text("ENABLE_LIVE_TRADING")
    monkeypatch.setattr(config, "LIVE_TRADING_TOKEN_PATH", str(token), raising=False)
    monkeypatch.setattr(config, "LIVE_TRADING_TOKEN_VALUE", "ENABLE_LIVE_TRADING", raising=False)

    token.chmod(0o777)
    with pytest.raises(permissions.PermissionError):
        permissions.ensure_open_trade_allowed()

    token.chmod(0o600)
    permissions.ensure_open_trade_allowed()
