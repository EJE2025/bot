"""Helpers to reconcile pending trades when the websocket desynchronises."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterable

from . import config, execution, trade_manager
from .state_machine import TradeState

logger = logging.getLogger(__name__)


def _state_from_trade(trade: dict) -> TradeState:
    try:
        return TradeState(trade.get("state"))
    except ValueError:
        return TradeState.PENDING


def reconcile_pending_trades(trades: Iterable[dict] | None = None) -> None:
    """Poll REST endpoints to promote or cancel stale pending trades."""

    now = time.time()
    active = list(trades) if trades is not None else trade_manager.all_open_trades()
    for trade in active:
        state = _state_from_trade(trade)
        if state != TradeState.PENDING:
            continue

        trade_id = trade.get("trade_id")
        symbol = trade.get("symbol", "")
        created = float(trade.get("created_ts") or trade.get("created_at") or now)
        age = now - created
        if age >= config.PENDING_FILL_TIMEOUT_S:
            logger.info(
                "Cancel PENDING by timeout: trade_id=%s symbol=%s age=%.1fs",
                trade_id,
                symbol,
                age,
            )
            order_id = trade.get("order_id")
            if order_id and not config.DRY_RUN:
                try:
                    execution.cancel_order(order_id, symbol)
                except Exception as exc:  # pragma: no cover - network errors
                    logger.debug(
                        "Cancel order %s failed during reconcile: %s", order_id, exc
                    )
            trade_manager.cancel_pending_trade(trade_id, reason="pending_timeout")
            continue

        if config.DRY_RUN:
            continue

        order_id = trade.get("order_id")
        if not order_id:
            continue

        status = execution.fetch_order_status(order_id, symbol)
        status = (status or "").lower()
        if status in {"filled"}:
            logger.info("Trade %s filled via REST fallback", trade_id)
            opened_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            trade_manager.update_trade(trade_id, open_time=opened_at)
            trade_manager.set_trade_state(trade_id, TradeState.OPEN)
        elif status in {"partial"}:
            logger.info("Trade %s partially filled via REST fallback", trade_id)
            trade_manager.set_trade_state(trade_id, TradeState.OPEN)
            trade_manager.set_trade_state(trade_id, TradeState.PARTIALLY_FILLED)
        elif status in {"canceled", "rejected", "expired"}:
            logger.info("Trade %s cancelled at exchange while pending", trade_id)
            trade_manager.cancel_pending_trade(trade_id, reason=status)
