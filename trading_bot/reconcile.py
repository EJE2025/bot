"""Helpers to reconcile pending trades when the websocket desynchronises."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterable

from . import config, execution, trade_manager
from .state_machine import TradeState
from .utils import normalize_symbol

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
            recovered = False
            if not config.DRY_RUN:
                norm_symbol = normalize_symbol(symbol)
                for pos in execution.fetch_positions():
                    pos_symbol = normalize_symbol(pos.get("symbol", ""))
                    try:
                        contracts = abs(float(pos.get("contracts", 0.0)))
                    except (TypeError, ValueError):
                        continue
                    if pos_symbol != norm_symbol or contracts <= 0:
                        continue

                    try:
                        entry_price = float(pos.get("entryPrice") or 0.0)
                    except (TypeError, ValueError):
                        entry_price = 0.0

                    logger.info(
                        "Recovered filled order for %s via positions; marking as OPEN",
                        symbol,
                    )
                    trade_manager.update_trade(
                        trade_id,
                        entry_price=entry_price,
                        quantity=contracts,
                        quantity_remaining=contracts,
                        status="active",
                        open_time=datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        order_id=pos.get("info", {}).get("orderId") or trade.get("order_id"),
                    )
                    trade_manager.set_trade_state(trade_id, TradeState.OPEN)
                    recovered = True
                    break

            if recovered:
                continue

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
        if status in {"filled", "partial"}:
            logger.info(
                "Trade %s shows %s via REST; awaiting position confirmation before state change",
                trade_id,
                status,
            )
        elif status in {"canceled", "rejected", "expired"}:
            logger.info(
                "Trade %s reported %s via REST; keeping pending until corroborated",
                trade_id,
                status,
            )
            try:
                execution.cancel_order(order_id, symbol)
            except Exception as exc:  # pragma: no cover - defensive network call
                logger.debug(
                    "Cancel order %s skipped during reconcile (status %s): %s",
                    order_id,
                    status,
                    exc,
                )
