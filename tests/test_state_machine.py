import unittest

from trading_bot.state_machine import (
    ALLOWED_TRANSITIONS,
    InvalidStateTransition,
    StatefulTrade,
    TradeState,
)


class TestStateMachine(unittest.TestCase):
    def test_valid_transitions_map_is_symmetric_with_rules(self) -> None:
        """Ensure every TradeState appears in the transitions map."""
        for state in TradeState:
            self.assertIn(state, ALLOWED_TRANSITIONS)

    def test_all_valid_transitions(self) -> None:
        """All allowed transitions should succeed."""
        for src, destinations in ALLOWED_TRANSITIONS.items():
            trade = StatefulTrade(state=src)
            for dst in destinations:
                with self.subTest(src=src, dst=dst):
                    trade.state = src  # reset to source state
                    trade.transition_to(dst)
                    self.assertEqual(trade.state, dst)

    def test_invalid_examples(self) -> None:
        """Selected invalid transitions should raise InvalidStateTransition."""
        invalid_pairs = [
            (TradeState.PENDING, TradeState.CLOSING),
            (TradeState.PENDING, TradeState.CLOSED),
            (TradeState.OPEN, TradeState.CLOSED),
            (TradeState.CLOSED, TradeState.OPEN),
            (TradeState.FAILED, TradeState.OPEN),
            (TradeState.CLOSING, TradeState.OPEN),
        ]
        for src, dst in invalid_pairs:
            with self.subTest(src=src, dst=dst):
                trade = StatefulTrade(state=src)
                with self.assertRaises(InvalidStateTransition):
                    trade.transition_to(dst)


if __name__ == "__main__":
    unittest.main()

