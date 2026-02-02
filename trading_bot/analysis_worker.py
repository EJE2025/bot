from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import time
import uuid
from typing import Any, Dict, Optional

import pandas as pd

from trading_bot import config, data
from trading_bot.analysis_store import AnalysisReport, GLOBAL_ANALYSIS_STORE
from trading_bot.charting import render_analysis_png
from trading_bot.patterns import build_technical_analysis
from trading_bot.ai_assistant import generate_llm_explanation

logger = logging.getLogger(__name__)


def _market_data_to_df(market: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(market)
    if "open" not in df.columns:
        df["open"] = df["close"].shift(1).fillna(df["close"])
    if "volume" not in df.columns:
        if "vol" in df.columns:
            df["volume"] = df["vol"]
        else:
            df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


@dataclass(slots=True)
class AnalysisJob:
    symbol: str
    interval: str
    reason: str


class AnalysisWorker:
    def __init__(self) -> None:
        self.q: "queue.Queue[AnalysisJob]" = queue.Queue(maxsize=200)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def enqueue(self, symbol: str, interval: Optional[str] = None, reason: str = "manual") -> bool:
        job = AnalysisJob(
            symbol=symbol,
            interval=interval or config.TECH_ANALYSIS_INTERVAL,
            reason=reason,
        )
        try:
            self.q.put_nowait(job)
            return True
        except queue.Full:
            logger.warning("Cola de análisis técnico llena, se omite %s", symbol)
            return False

    def _run(self) -> None:
        while True:
            job = self.q.get()
            try:
                market = data.get_market_data(
                    job.symbol, interval=job.interval, limit=config.TECH_ANALYSIS_LIMIT
                )
                if not market:
                    logger.warning("Sin datos de mercado para %s (%s)", job.symbol, job.interval)
                    continue
                df = _market_data_to_df(market)
                analysis = build_technical_analysis(job.symbol, job.interval, df)
                chart_path = render_analysis_png(df, analysis, out_dir="charts")

                llm_out = None
                if config.LLM_ENABLED:
                    llm_out = generate_llm_explanation(
                        analysis,
                        model=config.OPENAI_MODEL,
                        timeout=config.LLM_TIMEOUT_SECONDS,
                        max_tokens=config.LLM_MAX_TOKENS,
                    )

                report = AnalysisReport(
                    id=str(uuid.uuid4()),
                    created_at=time.time(),
                    symbol=job.symbol,
                    interval=job.interval,
                    analysis=analysis,
                    chart_path=chart_path,
                    llm_output=llm_out,
                    reason=job.reason,
                )
                GLOBAL_ANALYSIS_STORE.add(report)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Error generando análisis técnico: %s", exc)
            finally:
                self.q.task_done()


ANALYSIS_WORKER = AnalysisWorker()
