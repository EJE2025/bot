from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def render_analysis_png(df: pd.DataFrame, analysis: Dict[str, Any], out_dir: str = "charts") -> str:
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df.index = pd.RangeIndex(start=0, stop=len(plot_df), step=1)
    plot_df = plot_df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    symbol = analysis.get("symbol", "SYMBOL")
    interval = analysis.get("interval", "TF")
    filename = f"analysis_{symbol}_{interval}.png".replace("/", "_")
    out_path = str(Path(out_dir) / filename)

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        volume=False,
        returnfig=True,
        title=f"{symbol} · {interval}",
        style="yahoo",
    )
    ax = axes[0]

    for zone in analysis.get("support_zones", []):
        ax.axhspan(zone["from"], zone["to"], alpha=0.15, color="#22c55e")
    for zone in analysis.get("resistance_zones", []):
        ax.axhspan(zone["from"], zone["to"], alpha=0.15, color="#f97316")

    channel = analysis.get("channel")
    if channel:
        lookback = int(channel.get("lookback", len(plot_df)))
        start_idx = max(0, len(plot_df) - lookback)
        x_vals = list(range(start_idx, len(plot_df)))
        lower = channel["lower"]
        upper = channel["upper"]
        if x_vals:
            lower_line = [lower["y1"], lower["y2"]]
            upper_line = [upper["y1"], upper["y2"]]
            ax.plot([x_vals[0], x_vals[-1]], lower_line, color="#38bdf8", linewidth=1.2)
            ax.plot([x_vals[0], x_vals[-1]], upper_line, color="#38bdf8", linewidth=1.2)

    divs = analysis.get("rsi_divergences", [])
    if divs:
        last = divs[-1]
        idx = int(last["idx"])
        if idx < len(plot_df):
            y = float(plot_df["Close"].iloc[idx])
            color = "#22c55e" if last["type"] == "bullish" else "#f97316"
            ax.scatter([idx], [y], marker="o", color=color, s=40, zorder=5)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
