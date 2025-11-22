"""Utilities to train and save recurrent sequence models (LSTM/GRU).

The live strategy can already consume a sequential model via
``config.MODEL_SEQ_PATH``. This module provides a lightweight training helper
based on TensorFlow/Keras to create those models from CSV datasets containing
candlestick features.

Key assumptions
---------------
* Input columns must contain at least the OHLCV fields used at inference time
  (``["close", "high", "low", "vol"]`` by default). Additional numeric columns
  can be specified through ``feature_cols``.
* A binary ``target`` column encodes whether the trade outcome was profitable.
* Sequences are z-score normalised per-window in the same fashion as
  ``strategy._normalize_sequence_features``.

Example
-------
>>> from trading_bot.sequence_trainer import train_sequence_classifier
>>> model = train_sequence_classifier(
...     csv_path="trades.csv",
...     target="is_profitable",
...     window=32,
...     output_path="models/model_lstm.keras",
... )
>>> model.summary()
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_FEATURES = ["close", "high", "low", "vol"]


def _require_keras():
    spec = importlib.util.find_spec("tensorflow.keras")
    if spec is None:
        raise ImportError(
            "TensorFlow/Keras is required for sequence training. Install tensorflow>=2.12"
        )
    return importlib.import_module("tensorflow.keras")


def _windowed_zscore(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True)
    return (sequence - mean) / (std + 1e-8)


def build_sequence_dataset(
    frame: pd.DataFrame,
    target: str,
    *,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    window: int = 32,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a time-ordered frame into overlapping sequences and labels."""

    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")
    if len(frame) <= window:
        raise ValueError("Dataset is too short for the requested window size")

    values = frame[list(feature_cols)].to_numpy(dtype=float)
    labels = frame[target].to_numpy()

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for end_idx in range(window, len(frame), step):
        start_idx = end_idx - window
        seq = values[start_idx:end_idx]
        sequences.append(_windowed_zscore(seq))
        targets.append(labels[end_idx])

    return np.stack(sequences), np.asarray(targets, dtype=float)


def build_lstm_classifier(
    seq_len: int,
    num_features: int,
    *,
    units: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
):
    """Create a simple LSTM-based binary classifier using Keras."""

    keras = _require_keras()
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(seq_len, num_features)),
            keras.layers.LSTM(units, return_sequences=False),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(units // 2, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_sequence_classifier(
    csv_path: str | Path,
    target: str,
    *,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    window: int = 32,
    batch_size: int = 64,
    epochs: int = 20,
    val_split: float = 0.2,
    step: int = 1,
    output_path: str | Path | None = None,
):
    """Train an LSTM/GRU classifier from a CSV file of sequential data."""

    frame = pd.read_csv(csv_path)
    X, y = build_sequence_dataset(
        frame, target, feature_cols=feature_cols, window=window, step=step
    )

    model = build_lstm_classifier(window, X.shape[-1])
    model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        verbose=2,
        shuffle=False,
    )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sequential LSTM classifier for the trading bot",
    )
    parser.add_argument("csv", help="CSV with OHLCV features and target column")
    parser.add_argument("--target", default="target", help="Binary target column")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to include in each sequence (default: close high low vol)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=32,
        help="Number of candles per sequence window (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Stride between sequences (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="model_seq.keras",
        help="Path to save the trained model (default: %(default)s)",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    train_sequence_classifier(
        csv_path=args.csv,
        target=args.target,
        feature_cols=args.features,
        window=args.window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        step=args.step,
        output_path=args.output,
    )
    print(f"Sequential model saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
