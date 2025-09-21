import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot.latency import measure_latency
from trading_bot.metrics import LATENCY_HISTOGRAMS


def _histogram_count(histogram):
    for sample in histogram.collect()[0].samples:
        if sample.name.endswith("_count"):
            return sample.value
    return 0.0


def test_measure_latency_records_histogram():
    histogram = LATENCY_HISTOGRAMS["feature_to_prediction"]
    before = _histogram_count(histogram)
    with measure_latency("feature_to_prediction"):
        time.sleep(0.01)
    after = _histogram_count(histogram)
    assert after == before + 1
