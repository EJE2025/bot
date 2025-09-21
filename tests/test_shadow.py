import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot.shadow import (
    ShadowConfig,
    configure,
    record_shadow_signal,
    finalize_shadow_trade,
)


def test_shadow_records_to_disk(tmp_path):
    cfg = ShadowConfig(enabled=True, record_to=tmp_path)
    configure(cfg)
    record_shadow_signal("BTCUSDT", "heuristic", {"prob_success": 0.6})
    finalize_shadow_trade("trade-1", {"pnl": 1.5})
    signals_path = tmp_path / "signals.jsonl"
    results_path = tmp_path / "results.jsonl"
    assert signals_path.exists()
    assert results_path.exists()
    assert "BTCUSDT" in signals_path.read_text(encoding="utf-8")
    assert "trade-1" in results_path.read_text(encoding="utf-8")
