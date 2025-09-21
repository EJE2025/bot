import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import idempotency


def test_idempotency_prevents_duplicates():
    idempotency.reset_cache(ttl_seconds=1)
    key = idempotency.build_idempotency_key("BTCUSDT", "BUY", 100.0, 1.0, bucket_seconds=1)
    assert idempotency.should_submit(key) is True
    idempotency.store_result(key, {"id": "abc"})
    assert idempotency.get_cached_result(key) == {"id": "abc"}
    assert idempotency.should_submit(key) is False
    assert idempotency.get_cached_result(key) == {"id": "abc"}
    time.sleep(1.1)
    assert idempotency.should_submit(key) is True
    idempotency.reset_cache()
