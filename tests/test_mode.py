"""Tests for runtime mode selection utilities."""

from __future__ import annotations

import types

from trading_bot import mode


def test_resolve_mode_priority() -> None:
    assert mode.resolve_mode("shadow", "heuristic") == "shadow"
    assert mode.resolve_mode(None, "shadow") == "shadow"
    # Unknown modes fallback to normal
    assert mode.resolve_mode(None, "desconocido") == "normal"


def test_apply_mode_profiles() -> None:
    conf = types.SimpleNamespace(
        ENABLE_TRADING=True,
        SHADOW_MODE=False,
        ENABLE_MODEL=True,
        MODEL_WEIGHT=0.5,
        DRY_RUN=False,
        MAINTENANCE=False,
        RUN_BACKTEST_ON_START=False,
    )

    mode.apply_mode_to_config("shadow", conf)
    assert conf.ENABLE_TRADING is False
    assert conf.SHADOW_MODE is True
    assert conf.ENABLE_MODEL is True
    assert conf.DRY_RUN is False
    assert conf.MAINTENANCE is False
    assert conf.RUN_BACKTEST_ON_START is False

    mode.apply_mode_to_config("heuristic", conf)
    assert conf.ENABLE_TRADING is True
    assert conf.SHADOW_MODE is False
    assert conf.ENABLE_MODEL is False
    assert conf.MODEL_WEIGHT == 0.0


def test_interactive_pick_without_tty(monkeypatch) -> None:
    dummy = types.SimpleNamespace(isatty=lambda: False)
    monkeypatch.setattr(mode, "sys", types.SimpleNamespace(stdin=dummy))
    assert mode.interactive_pick() == "normal"
