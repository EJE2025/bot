from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from trading_bot import config, exporter


def test_append_trade_closed_preserves_existing_dates(tmp_path, monkeypatch):
    excel_dir = tmp_path / "exports"
    excel_dir.mkdir()

    monkeypatch.setattr(config, "EXPORTS_DIR", str(excel_dir))
    monkeypatch.setattr(config, "EXCEL_TRADES", "trades_closed.xlsx")

    first_trade = {
        "symbol": "BTCUSDT",
        "side": "buy",
        "entry_price": 100,
        "exit_price": 110,
        "quantity": 1,
        "open_time": "2024-01-01T00:00:00Z",
        "close_time": "2024-01-02T00:00:00Z",
        "status": "closed",
        "trade_id": "t1",
        "close_reason": "test",
    }

    second_trade = {
        "symbol": "ETHUSDT",
        "side": "buy",
        "entry_price": 200,
        "exit_price": 210,
        "quantity": 1,
        "open_time": "2024-02-01 00:00:00",  # No timezone information
        "close_time": "2024-02-02T00:00:00Z",  # Explicit UTC
        "status": "closed",
        "trade_id": "t2",
        "close_reason": "test",
    }

    exporter.append_trade_closed(first_trade)
    exporter.append_trade_closed(second_trade)

    df = pd.read_excel(excel_dir / "trades_closed.xlsx", sheet_name="trades")

    assert list(df["open_time"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert list(df["close_time"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-02")]
