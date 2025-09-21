import math
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot.exchange_rules import (
    SymbolRules,
    quantize_price,
    quantize_qty,
    validate_order,
    ValidationError,
)


def test_quantize_price_and_qty():
    rules = SymbolRules(price_tick=0.05, qty_step=0.1, min_qty=0.1)
    assert math.isclose(quantize_price(100.07, rules.price_tick), 100.05)
    assert math.isclose(quantize_qty(0.37, rules.qty_step), 0.3)


def test_validate_order_checks_minimums():
    rules = SymbolRules(price_tick=0.01, qty_step=0.1, min_qty=1.0, min_notional=10.0)
    validate_order("BTCUSDT", "BUY", 100.0, 1.0, rules)
    with pytest.raises(ValidationError):
        validate_order("BTCUSDT", "BUY", 100.0, 0.5, rules)
    with pytest.raises(ValidationError):
        validate_order("BTCUSDT", "BUY", -1.0, 1.0, rules)
    with pytest.raises(ValidationError):
        validate_order("BTCUSDT", "BUY", 5.0, 1.0, rules)
