import os
import sys
import json
import requests
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import data


def test_top_liquidity_levels_none():
    bids, asks = data.top_liquidity_levels(None)
    assert bids == [] and asks == []


def test_get_order_book_invalid_symbol(monkeypatch):
    resp = requests.Response()
    resp.status_code = 400
    resp._content = b'{"msg":"Invalid symbol"}'
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: resp)
    monkeypatch.setattr(data.time, "sleep", lambda x: None)
    book = data.get_order_book("PEPE_USDT")
    assert book is None
