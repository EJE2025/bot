import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trading_bot import strategy


def test_calcular_tamano_atr_zero():
    size = strategy.calcular_tamano_posicion(
        balance_usdt=1000,
        entry_price=100,
        atr_value=0,
        atr_multiplier=1.5,
        risk_per_trade_usd=10,
    )
    assert size is None


def test_calcular_tamano_distancia_negativa():
    size = strategy.calcular_tamano_posicion(
        balance_usdt=1000,
        entry_price=100,
        atr_value=1,
        atr_multiplier=-1,
        risk_per_trade_usd=10,
    )
    assert size is None


def test_calcular_tamano_capped_by_balance():
    size = strategy.calcular_tamano_posicion(
        balance_usdt=100,
        entry_price=10,
        atr_value=0.5,
        atr_multiplier=1,
        risk_per_trade_usd=1000,
    )
    # Available balance only allows 10 contracts
    assert size == pytest.approx(10)


def test_position_sizer_fixed_shadow(monkeypatch):
    monkeypatch.setattr(strategy.config, "USE_FIXED_POSITION_SIZE", True)
    monkeypatch.setattr(strategy.config, "FIXED_POSITION_SIZE_USDT", 20.0)
    monkeypatch.setattr(strategy.config, "MIN_POSITION_SIZE_USDT", 5.0)
    monkeypatch.setattr(strategy.config, "BOT_MODE", "shadow")

    features = {"entry_price": 100.0, "atr": 1.0, "atr_multiplier": 1.0}
    size = strategy.position_sizer("BTC_USDT", features, {})
    assert size == pytest.approx(20.0)


def test_position_sizer_shadow_minimum(monkeypatch):
    monkeypatch.setattr(strategy.config, "USE_FIXED_POSITION_SIZE", True)
    monkeypatch.setattr(strategy.config, "FIXED_POSITION_SIZE_USDT", 2.0)
    monkeypatch.setattr(strategy.config, "MIN_POSITION_SIZE_USDT", 5.0)
    monkeypatch.setattr(strategy.config, "BOT_MODE", "shadow")

    features = {"entry_price": 100.0, "atr": 1.0, "atr_multiplier": 1.0}
    size = strategy.position_sizer("BTC_USDT", features, {})
    assert size == pytest.approx(5.0)

