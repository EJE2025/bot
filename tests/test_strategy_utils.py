import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import config
from trading_bot.strategy import calcular_tamano_posicion, risk_reward_ratio


def test_calcular_tamano_posicion_invalid():
    qty = calcular_tamano_posicion(1000, 10, 0, 1, 100)
    assert qty is None


def test_calcular_tamano_posicion_minimum():
    qty = calcular_tamano_posicion(1000, 10, 1, 1, 10)
    assert (qty is None) or qty >= config.MIN_POSITION_SIZE


def test_risk_reward_ratio():
    ratio = risk_reward_ratio(100, 110, 95)
    assert ratio == (10)/(5)
    assert risk_reward_ratio(100, 90, 105) == (10)/(5)
    assert risk_reward_ratio(100, 110, 100) == 0.0
