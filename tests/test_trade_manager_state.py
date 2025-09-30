import unittest
from trading_bot import trade_manager as tm
from trading_bot.state_machine import TradeState


class TestTradeManagerWithState(unittest.TestCase):
    def setUp(self):
        if hasattr(tm, "reset_state"):
            tm.reset_state()

    def test_happy_path(self):
        tm.add_trade({"trade_id": "T1", "symbol": "BTC_USDT"})
        tr = tm.find_trade(trade_id="T1")
        self.assertIsNotNone(tr)
        self.assertEqual(tr["state"], TradeState.PENDING.value)

        tm.set_trade_state("T1", TradeState.OPEN)
        self.assertEqual(tm.find_trade(trade_id="T1")["state"], TradeState.OPEN.value)

        tm.set_trade_state("T1", TradeState.PARTIALLY_FILLED)
        self.assertEqual(
            tm.find_trade(trade_id="T1")["state"], TradeState.PARTIALLY_FILLED.value
        )

        tm.set_trade_state("T1", TradeState.OPEN)
        tm.set_trade_state("T1", TradeState.CLOSING)

        closed = tm.close_trade("T1", exit_price=100.0, profit=5.0)
        self.assertEqual(closed["state"], TradeState.CLOSED.value)

    def test_invalid_transition(self):
        tm.add_trade({"trade_id": "T2", "symbol": "ETH_USDT"})
        result = tm.set_trade_state("T2", TradeState.CLOSED)
        self.assertFalse(result)
        self.assertEqual(
            tm.find_trade(trade_id="T2")["state"], TradeState.PENDING.value
        )


if __name__ == "__main__":
    unittest.main()
