#!/usr/bin/env python3
"""
Utility script to prepare and run a backtest for the trading bot.

This script expects that the `trading_bot` package is available in your
Python environment. It will load the configuration from `backtest.yml`
and the dataset from `backtest.csv` located in the current working
directory, run the backtester, and print the resulting KPIs.

If the `trading_bot` package is not installed, you can instead invoke
`python -m trading_bot.backtest --config backtest.yml --data backtest.csv`
manually after generating the YAML and CSV files in your working directory.
"""

import json
import sys
from pathlib import Path

# Define file paths relative to the script location
cfg_path = Path(__file__).with_name('backtest.yml')
data_path = Path(__file__).with_name('backtest.csv')

if not cfg_path.exists():
    sys.exit(f"Backtest config not found: {cfg_path}")
if not data_path.exists():
    sys.exit(f"Backtest dataset not found: {data_path}")

try:
    from trading_bot import backtest
except ImportError:
    import subprocess
    # Fallback: run the module via subprocess
    print("`trading_bot` package not found; falling back to subprocess execution.")
    subprocess.run([
        sys.executable,
        '-m', 'trading_bot.backtest',
        '--config', str(cfg_path),
        '--data', str(data_path),
    ], check=True)
    sys.exit(0)

# If trading_bot was imported, run backtest directly
cfg = backtest._load_config(cfg_path)
data = backtest._load_dataset(data_path)
results = backtest.run_backtest(cfg, data)
print(json.dumps(results, indent=2))
