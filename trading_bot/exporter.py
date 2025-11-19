"""Utilities for exporting bot state snapshots to Excel workbooks."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from filelock import FileLock

from . import config


def _ensure_dir(directory: Path) -> None:
    """Ensure that ``directory`` exists."""
    directory.mkdir(parents=True, exist_ok=True)


def _excel_path(name: str) -> Path:
    """Return an absolute path for the Excel workbook respecting config."""
    candidate = Path(name)
    if candidate.is_absolute():
        _ensure_dir(candidate.parent)
        return candidate

    base_dir = Path(config.EXPORTS_DIR)
    _ensure_dir(base_dir)
    target = base_dir / candidate
    _ensure_dir(target.parent)
    return target


def _remove_timezone(series: pd.Series) -> pd.Series:
    """Return ``series`` converted to naive datetimes when possible."""
    try:
        converted = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (ValueError, TypeError):
        return pd.Series(pd.NaT, index=series.index)

    try:
        return converted.dt.tz_convert(None)
    except (AttributeError, TypeError, ValueError):
        try:
            return converted.dt.tz_localize(None)
        except (AttributeError, TypeError, ValueError):
            return converted


def _to_numeric(series: pd.Series) -> pd.Series:
    """Return ``series`` coerced to numeric values when possible."""
    return pd.to_numeric(series, errors="coerce")


def _normalize_dataframe(
    df: pd.DataFrame,
    *,
    date_cols: Iterable[str] = (),
    numeric_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Return a copy of ``df`` with requested columns normalised."""

    df_norm = df.copy()
    for column in date_cols:
        if column in df_norm.columns:
            df_norm[column] = _remove_timezone(df_norm[column])

    for column in numeric_cols:
        if column in df_norm.columns:
            df_norm[column] = _to_numeric(df_norm[column])

    return df_norm


def _write_workbook(path: Path, sheet_specs: Sequence[Dict[str, Any]]) -> None:
    """Write ``sheet_specs`` to ``path`` applying rich formatting."""

    if not sheet_specs:
        return

    _ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with pd.ExcelWriter(
        tmp_path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd hh:mm",
        date_format="yyyy-mm-dd hh:mm",
    ) as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#F2F2F2",
            "border": 1,
        })
        money_fmt = workbook.add_format({"num_format": "#,##0.00", "border": 1})
        pct_fmt = workbook.add_format({"num_format": "0.00%", "border": 1})
        int_fmt = workbook.add_format({"num_format": "0", "border": 1})
        float_fmt = workbook.add_format({"num_format": "#,##0.0000", "border": 1})
        text_fmt = workbook.add_format({"border": 1})
        date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm", "border": 1})

        for spec in sheet_specs:
            sheet_name = spec["sheet_name"]
            dataframe = spec.get("dataframe")
            if dataframe is None:
                dataframe = pd.DataFrame()

            date_cols: Sequence[str] = spec.get("date_cols", ())
            numeric_cols: Sequence[str] = spec.get("numeric_cols", ())

            df_norm = _normalize_dataframe(
                dataframe,
                date_cols=date_cols,
                numeric_cols=set(numeric_cols)
                | set(spec.get("money_cols", ()))
                | set(spec.get("pct_cols", ()))
                | set(spec.get("int_cols", ())),
            )

            df_norm.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            nrows, ncols = df_norm.shape
            if ncols == 0:
                worksheet.write(0, 0, "No data available", header_fmt)
                continue

            money_cols = set(spec.get("money_cols", ()))
            pct_cols = set(spec.get("pct_cols", ()))
            int_cols = set(spec.get("int_cols", ()))
            number_cols = set(spec.get("numeric_cols", ()))

            column_widths: Dict[str, int] = {
                column: max(12, min(40, len(column) + 2))
                for column in df_norm.columns
            }
            column_widths.update(spec.get("column_widths", {}))

            table_columns = []
            for idx, column in enumerate(df_norm.columns):
                width = column_widths.get(column, 12)
                worksheet.set_column(idx, idx, width)

                if column in money_cols:
                    fmt = money_fmt
                elif column in pct_cols:
                    fmt = pct_fmt
                elif column in int_cols:
                    fmt = int_fmt
                elif column in number_cols or pd.api.types.is_numeric_dtype(df_norm[column]):
                    fmt = float_fmt
                elif column in date_cols:
                    fmt = date_fmt
                else:
                    fmt = text_fmt

                table_columns.append({"header": column, "format": fmt})

            worksheet.add_table(
                0,
                0,
                max(nrows, 0),
                ncols - 1,
                {
                    "style": spec.get("table_style", "Table Style Medium 9"),
                    "columns": table_columns,
                    "autofilter": True,
                },
            )

            worksheet.freeze_panes(
                spec.get("freeze_rows", 1), spec.get("freeze_cols", 0)
            )

            conditional_pnl_col = spec.get("conditional_pnl_col")
            if conditional_pnl_col and conditional_pnl_col in df_norm.columns and nrows > 0:
                col_idx = df_norm.columns.get_loc(conditional_pnl_col)
                worksheet.conditional_format(
                    1,
                    col_idx,
                    nrows,
                    col_idx,
                    {
                        "type": "3_color_scale",
                        "min_color": "#FF736A",
                        "mid_color": "#FFFFFF",
                        "max_color": "#A8E6A1",
                    },
                )

    os.replace(tmp_path, path)


def append_trade_closed(
    trade: Dict[str, Any], *, extra: Optional[Dict[str, Any]] = None
) -> Path:
    """Persist a closed trade into the configured workbook."""

    def _first_not_none(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    row: Dict[str, Any] = {
        "closed_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "symbol": trade.get("symbol"),
        "side": trade.get("side"),
        "entry_price": trade.get("entry_price"),
        "exit_price": _first_not_none(trade.get("exit_price"), trade.get("price")),
        "quantity": trade.get("quantity"),
        "pnl": _first_not_none(trade.get("pnl"), trade.get("profit"), 0.0),
        "rr": trade.get("risk_reward"),
        "stop_loss": trade.get("stop_loss"),
        "take_profit": trade.get("take_profit"),
        "open_time": _first_not_none(
            trade.get("open_time"),
            trade.get("open_time_dt"),
            trade.get("opened_at"),
            trade.get("created_at"),
        ),
        "close_time": _first_not_none(
            trade.get("close_time"),
            trade.get("close_time_dt"),
            trade.get("closed_at"),
            trade.get("updated_at"),
        ),
        "timeframe": trade.get("timeframe"),
        "volatility": trade.get("volatility"),
        "prob_success": trade.get("prob_success"),
        "orig_prob": trade.get("orig_prob"),
        "prob_threshold": trade.get("prob_threshold"),
        "leverage": trade.get("leverage"),
        "status": trade.get("status"),
        "trade_id": trade.get("trade_id"),
        "close_reason": trade.get("close_reason"),
    }

    snapshot = trade.get("feature_snapshot") or {}
    if isinstance(snapshot, dict):
        for key, value in snapshot.items():
            row[f"feat_{key}"] = value

    if extra:
        for key, value in extra.items():
            row[f"extra_{key}"] = value

    path = _excel_path(config.EXCEL_TRADES)
    lock = FileLock(str(path) + ".lock")
    with lock:
        existing = pd.DataFrame()
        if path.exists():
            try:
                existing = pd.read_excel(path, sheet_name="trades")
            except Exception:
                existing = pd.DataFrame()

        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)

        _write_workbook(
            path,
            [
                {
                    "sheet_name": "trades",
                    "dataframe": updated,
                    "date_cols": ["closed_at", "open_time", "close_time"],
                    "money_cols": [
                        "entry_price",
                        "exit_price",
                        "stop_loss",
                        "take_profit",
                        "pnl",
                    ],
                    "pct_cols": ["prob_success", "orig_prob", "prob_threshold"],
                    "int_cols": ["leverage"],
                    "numeric_cols": [
                        "quantity",
                        "rr",
                        "volatility",
                        "prob_success",
                        "orig_prob",
                        "prob_threshold",
                    ],
                    "column_widths": {
                        "symbol": 12,
                        "side": 8,
                        "entry_price": 12,
                        "exit_price": 12,
                        "quantity": 12,
                        "pnl": 12,
                        "rr": 8,
                        "timeframe": 12,
                        "volatility": 12,
                        "prob_success": 12,
                        "orig_prob": 12,
                        "prob_threshold": 14,
                        "trade_id": 36,
                        "close_reason": 20,
                    },
                    "conditional_pnl_col": "pnl",
                }
            ],
        )
    return path


def write_ai_status(
    *,
    model_info: Dict[str, Any],
    training_metrics: Dict[str, Any],
    runtime_metrics: Dict[str, Any],
) -> Path:
    """Write the latest AI status snapshot overwriting the workbook."""
    path = _excel_path(config.EXCEL_AI)
    lock = FileLock(str(path) + ".lock")
    with lock:
        model_df = pd.DataFrame([model_info or {}])
        training_df = pd.DataFrame([training_metrics or {}])
        runtime_df = pd.DataFrame([runtime_metrics or {}])

        _write_workbook(
            path,
            [
                {
                    "sheet_name": "model",
                    "dataframe": model_df,
                    "date_cols": ["mtime", "deployed_at"],
                    "numeric_cols": ["model_weight"],
                    "column_widths": {
                        column: max(14, min(48, len(column) + 2))
                        for column in model_df.columns
                    },
                },
                {
                    "sheet_name": "training",
                    "dataframe": training_df,
                    "date_cols": ["started_at", "completed_at"],
                    "numeric_cols": ["samples", "duration_s"],
                    "column_widths": {
                        column: max(14, min(48, len(column) + 2))
                        for column in training_df.columns
                    },
                },
                {
                    "sheet_name": "runtime",
                    "dataframe": runtime_df,
                    "numeric_cols": [
                        column
                        for column in runtime_df.columns
                        if pd.api.types.is_numeric_dtype(runtime_df[column])
                    ],
                    "column_widths": {
                        column: max(14, min(48, len(column) + 2))
                        for column in runtime_df.columns
                    },
                },
            ],
        )
    return path


def write_ops_snapshot(
    *,
    positions: List[Dict[str, Any]],
    risk_limits: Dict[str, Any],
    ws_status: Dict[str, Any],
) -> Path:
    """Write the current operations snapshot overwriting the workbook."""
    path = _excel_path(config.EXCEL_OPS)
    lock = FileLock(str(path) + ".lock")
    with lock:
        positions_df = pd.DataFrame(positions or [])
        risk_df = pd.DataFrame([risk_limits or {}])
        ws_df = pd.DataFrame([ws_status or {}])

        _write_workbook(
            path,
            [
                {
                    "sheet_name": "positions",
                    "dataframe": positions_df,
                    "date_cols": ["open_time", "close_time"],
                    "money_cols": [
                        "entry_price",
                        "stop_loss",
                        "take_profit",
                        "pnl",
                    ],
                    "pct_cols": ["prob_success"],
                    "int_cols": ["leverage"],
                    "numeric_cols": [
                        column
                        for column in [
                            "quantity",
                            "leverage",
                            "pnl",
                            "entry_price",
                            "stop_loss",
                            "take_profit",
                            "prob_success",
                        ]
                        if column in positions_df.columns
                    ],
                    "column_widths": {
                        "symbol": 12,
                        "side": 8,
                        "trade_id": 36,
                    },
                },
                {
                    "sheet_name": "risk_limits",
                    "dataframe": risk_df,
                    "numeric_cols": [
                        column
                        for column in risk_df.columns
                        if pd.api.types.is_numeric_dtype(risk_df[column])
                    ],
                    "column_widths": {
                        column: max(14, min(48, len(column) + 2))
                        for column in risk_df.columns
                    },
                },
                {
                    "sheet_name": "ws_status",
                    "dataframe": ws_df,
                    "date_cols": ["last_heartbeat"],
                    "numeric_cols": [
                        column
                        for column in ws_df.columns
                        if pd.api.types.is_numeric_dtype(ws_df[column])
                    ],
                    "column_widths": {
                        column: max(14, min(48, len(column) + 2))
                        for column in ws_df.columns
                    },
                },
            ],
        )
    return path
