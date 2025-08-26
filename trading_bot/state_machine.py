from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set
import uuid


class TradeState(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


# Transiciones permitidas:
# PENDING → OPEN / FAILED
# OPEN → PARTIALLY_FILLED / CLOSING / FAILED
# PARTIALLY_FILLED → OPEN / CLOSING / FAILED
# CLOSING → CLOSED / FAILED
# CLOSED → (ninguna)
# FAILED → (ninguna)
ALLOWED_TRANSITIONS: Dict[TradeState, Set[TradeState]] = {
    TradeState.PENDING: {TradeState.OPEN, TradeState.FAILED},
    TradeState.OPEN: {
        TradeState.PARTIALLY_FILLED,
        TradeState.CLOSING,
        TradeState.FAILED,
    },
    TradeState.PARTIALLY_FILLED: {
        TradeState.OPEN,
        TradeState.CLOSING,
        TradeState.FAILED,
    },
    TradeState.CLOSING: {TradeState.CLOSED, TradeState.FAILED},
    TradeState.CLOSED: set(),
    TradeState.FAILED: set(),
}


def is_valid_transition(src: TradeState, dst: TradeState) -> bool:
    return dst in ALLOWED_TRANSITIONS.get(src, set())


class InvalidStateTransition(Exception):
    pass


@dataclass
class StatefulTrade:
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TradeState = TradeState.PENDING
    # opcional: metadata mínima
    symbol: str | None = None
    side: str | None = None

    def can_transition_to(self, new_state: TradeState) -> bool:
        return is_valid_transition(self.state, new_state)

    def transition_to(self, new_state: TradeState) -> None:
        if not self.can_transition_to(new_state):
            msg = f"{self.state} → {new_state} no permitido"
            raise InvalidStateTransition(msg)
        self.state = new_state
