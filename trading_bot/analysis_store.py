from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import threading
from typing import Any, Dict, List, Optional

from trading_bot import config


@dataclass(frozen=True)
class AnalysisReport:
    id: str
    created_at: float
    symbol: str
    interval: str
    analysis: Dict[str, Any]
    chart_path: Optional[str] = None
    llm_output: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class AnalysisStore:
    def __init__(self, max_items: int = 200) -> None:
        self._lock = threading.Lock()
        self._items: deque[AnalysisReport] = deque(maxlen=max_items)

    def add(self, report: AnalysisReport) -> None:
        with self._lock:
            self._items.appendleft(report)

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(r) for r in list(self._items)[:limit]]

    def latest_for(self, symbol: str, interval: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            for report in self._items:
                if report.symbol == symbol and (interval is None or report.interval == interval):
                    return asdict(report)
        return None


GLOBAL_ANALYSIS_STORE = AnalysisStore(max_items=config.TECH_ANALYSIS_MAX_REPORTS)
