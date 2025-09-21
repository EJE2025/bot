import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import shutdown


def test_shutdown_callbacks_executed():
    shutdown.reset_for_tests()
    called = []

    def _callback():
        called.append(True)

    shutdown.register_callback(_callback)
    shutdown.request_shutdown()
    assert shutdown.shutdown_requested() is True
    shutdown.execute_callbacks()
    assert called == [True]
    shutdown.reset_for_tests()
