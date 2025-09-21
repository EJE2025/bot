"""Prometheus metrics and alerting helpers for the trading bot."""

from __future__ import annotations

import time
from typing import Dict

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

from . import notify

open_trades_gauge = Gauge(
    "trading_bot_open_trades",
    "Number of open trades",
)
closed_trades_gauge = Gauge(
    "trading_bot_closed_trades",
    "Number of closed trades",
)
api_latency = Summary(
    "trading_bot_api_latency_seconds",
    "Latency of API calls",
)

LATENCY_HISTOGRAMS: Dict[str, Histogram] = {
    "feature_to_prediction": Histogram(
        "trading_bot_latency_ms_feature_to_prediction",
        "Feature to prediction latency in milliseconds",
        buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    ),
    "submit_to_ack": Histogram(
        "trading_bot_latency_ms_submit_to_ack",
        "Order submission to acknowledgement latency in milliseconds",
        buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    ),
    "ack_to_filled": Histogram(
        "trading_bot_latency_ms_ack_to_filled",
        "Order acknowledgement to filled latency in milliseconds",
        buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    ),
}

model_hit_rate_gauge = Gauge(
    "trading_bot_model_hit_rate_rolling",
    "Rolling hit rate of the predictive model",
)
model_confidence_gauge = Gauge(
    "trading_bot_model_confidence_mean",
    "Average model confidence over the rolling window",
)
drift_score_gauge = Gauge(
    "trading_bot_model_drift_score",
    "Absolute drift between predicted probability and outcomes",
)
api_error_counter = Counter(
    "trading_bot_api_errors_total",
    "Total number of API errors detected",
)
api_retry_counter = Counter(
    "trading_bot_api_retries_total",
    "Total number of API retries executed",
)

_alert_cooldown_seconds = 300
_last_alert: Dict[str, float] = {}


def update_trade_metrics(open_count: int, closed_count: int) -> None:
    """Update open and closed trade gauges."""

    open_trades_gauge.set(open_count)
    closed_trades_gauge.set(closed_count)


def start_metrics_server(port: int = 8001) -> None:
    """Start an HTTP server exposing Prometheus metrics."""

    start_http_server(port)


def record_model_performance(hit_rate: float, confidence: float, drift: float) -> None:
    model_hit_rate_gauge.set(hit_rate)
    model_confidence_gauge.set(confidence)
    drift_score_gauge.set(drift)


def record_api_error() -> None:
    api_error_counter.inc()


def record_api_retry() -> None:
    api_retry_counter.inc()


def maybe_alert(condition: bool, message: str, *, cooldown: int | None = None) -> None:
    if not condition:
        return
    now = time.time()
    window = cooldown or _alert_cooldown_seconds
    last = _last_alert.get(message)
    if last and now - last < window:
        return
    _last_alert[message] = now
    notify.send_telegram(message)
    notify.send_discord(message)


__all__ = [
    "LATENCY_HISTOGRAMS",
    "api_latency",
    "record_model_performance",
    "record_api_error",
    "record_api_retry",
    "maybe_alert",
    "start_metrics_server",
    "update_trade_metrics",
]
