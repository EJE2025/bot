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


def test_shadow_integration_records_results(monkeypatch):
    from trading_bot import bot

    bot._reset_shadow_positions()
    monkeypatch.setattr(bot.config, "SHADOW_MODE", True)

    recorded: list[tuple[str, str, dict]] = []
    finalized: list[tuple[str, dict]] = []

    def fake_record(symbol, mode, payload):
        recorded.append((symbol, mode, payload))

    def fake_finalize(trade_id, payload):
        finalized.append((trade_id, payload))

    monkeypatch.setattr(bot.shadow, "record_shadow_signal", fake_record)
    monkeypatch.setattr(bot.shadow, "finalize_shadow_trade", fake_finalize)

    signal = {
        "symbol": "AAA_USDT",
        "side": "BUY",
        "quantity": 1.0,
        "entry_price": 100.0,
        "take_profit": 110.0,
        "stop_loss": 95.0,
        "prob_success": 0.7,
        "orig_prob": 0.55,
        "leverage": 1,
    }

    bot.open_new_trade(signal)
    assert len(recorded) == 2
    trade_ids = {payload["trade_id"] for _, _, payload in recorded}
    assert all(":" in tid for tid in trade_ids)

    monkeypatch.setattr(bot.data, "get_current_price_ticker", lambda symbol: 111.0)
    profit = bot._process_shadow_positions()

    assert profit > 0
    assert len(finalized) == 2
    modes = {payload["mode"] for _, payload in finalized}
    assert modes == {"heuristic", "hybrid"}
    bot._reset_shadow_positions()
