import csv
from pathlib import Path

HISTORY_FILE = Path("trade_history.csv")

FIELDS = [
    "symbol",
    "side",
    "quantity",
    "entry_price",
    "exit_price",
    "take_profit",
    "stop_loss",
    "profit",
    "open_time",
    "close_time",
]

def append_trade(row: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_FILE.exists()
    with HISTORY_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})

