"""Latency instrumentation utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager

from . import config
from .metrics import LATENCY_HISTOGRAMS


@contextmanager
def measure_latency(stage: str):
    """Context manager that records elapsed milliseconds for ``stage``."""

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        histogram = LATENCY_HISTOGRAMS.get(stage)
        if histogram is not None:
            histogram.observe(elapsed_ms)
        if elapsed_ms > config.LATENCY_SLO_MS:
            from .metrics import maybe_alert

            maybe_alert(True, f"Latency SLO exceeded for {stage}: {elapsed_ms:.1f} ms")
