"""Backward-compatible wrappers around :mod:`trading_bot.predictive_model`.

The legacy optimizer module previously exposed a lightweight Random Forest
training routine. To avoid duplicating the more feature-complete training
workflow in :mod:`trading_bot.predictive_model`, these helpers now delegate to
that module while issuing a ``DeprecationWarning`` so callers can migrate.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from . import predictive_model
from .predictive_model import TrainingConfig


def optimize_model(
    data: pd.DataFrame | str,
    target: str,
    *,
    config: Optional[TrainingConfig] = None,
):
    """Train a model by delegating to :mod:`predictive_model`.

    Parameters
    ----------
    data:
        Either a Pandas :class:`~pandas.DataFrame` with the training dataset or a
        path to a CSV file. Passing a dataframe mirrors the historical
        behaviour of this function.
    target:
        Name of the binary target column.
    config:
        Optional :class:`TrainingConfig` instance controlling training
        behaviour.
    """

    warnings.warn(
        "trading_bot.optimizer is deprecated; use trading_bot.predictive_model",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        return predictive_model.train_model_from_frame(frame, target, config)

    return predictive_model.train_model(str(data), target, config)


def save_model(model, path: str = "model.pkl") -> None:
    """Persist ``model`` to ``path`` using :func:`predictive_model.save_model`."""

    warnings.warn(
        "optimizer.save_model is deprecated; use predictive_model.save_model",
        DeprecationWarning,
        stacklevel=2,
    )
    predictive_model.save_model(model, path)


def load_model(path: str = "model.pkl"):
    """Load a model artefact stored at ``path``."""

    warnings.warn(
        "optimizer.load_model is deprecated; use predictive_model.load_model",
        DeprecationWarning,
        stacklevel=2,
    )
    return predictive_model.load_model(path)
