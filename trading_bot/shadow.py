"""Shadow trading utilities to benchmark strategies without submitting orders."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

from . import config


@dataclass
class ShadowConfig:
    enabled: bool = False
    compare_modes: Iterable[str] = field(default_factory=lambda: ("heuristic", "hybrid"))
    record_to: Path = Path("shadow_trades")


class ShadowRecorder:
    def __init__(self, cfg: ShadowConfig) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self.base = Path(cfg.record_to)
        if cfg.enabled:
            self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, suffix: str) -> Path:
        return self.base / f"{suffix}.jsonl"

    def record(self, suffix: str, payload: Dict) -> None:
        if not self.cfg.enabled:
            return
        with self._lock:
            self.base.mkdir(parents=True, exist_ok=True)
            path = self._path(suffix)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


_recorder = ShadowRecorder(
    ShadowConfig(enabled=config.SHADOW_MODE, compare_modes=("heuristic", "hybrid"), record_to=Path("shadow_trades"))
)


def configure(cfg: ShadowConfig) -> None:
    global _recorder
    _recorder = ShadowRecorder(cfg)


def record_shadow_signal(symbol: str, mode: str, signal_dict: Dict) -> None:
    payload = {"symbol": symbol, "mode": mode, **signal_dict}
    _recorder.record("signals", payload)


def finalize_shadow_trade(trade_id: str, result_dict: Dict) -> None:
    payload = {"trade_id": trade_id, **result_dict}
    _recorder.record("results", payload)
