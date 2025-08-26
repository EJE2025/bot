from prometheus_client import start_http_server, Gauge, Summary

# Basic bot metrics
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


def update_trade_metrics(open_count: int, closed_count: int) -> None:
    """Update open and closed trade gauges."""
    open_trades_gauge.set(open_count)
    closed_trades_gauge.set(closed_count)


def start_metrics_server(port: int = 8001) -> None:
    """Start an HTTP server exposing Prometheus metrics."""
    start_http_server(port)
