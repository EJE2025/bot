import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import config, strategy


@pytest.fixture(autouse=True)
def _reset_override():
    strategy.set_model_weight_override(None)
    yield
    strategy.set_model_weight_override(None)


def test_probability_threshold_respects_margin(monkeypatch):
    monkeypatch.setattr(config, "MIN_PROB_SUCCESS", 0.4, raising=False)
    monkeypatch.setattr(config, "PROBABILITY_MARGIN", 0.1, raising=False)
    monkeypatch.setattr(config, "FEE_EST", 0.2, raising=False)

    threshold = strategy.probability_threshold(1.0)
    breakeven = 0.2 / (1.0 + 0.2)
    assert pytest.approx(threshold, rel=1e-6) == max(0.4, breakeven + 0.1)


def test_probability_threshold_caps(monkeypatch):
    monkeypatch.setattr(config, "MIN_PROB_SUCCESS", 0.99, raising=False)
    monkeypatch.setattr(config, "PROBABILITY_MARGIN", 0.1, raising=False)
    monkeypatch.setattr(config, "FEE_EST", 0.5, raising=False)

    threshold = strategy.probability_threshold(0.1)
    assert threshold <= 0.995


def test_model_weight_override():
    base_weight = strategy.get_model_weight()
    strategy.set_model_weight_override(0.2)
    assert strategy.get_model_weight() == pytest.approx(0.2)
    strategy.set_model_weight_override(None)
    assert strategy.get_model_weight() == pytest.approx(base_weight)
