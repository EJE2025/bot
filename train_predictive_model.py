"""Command line utility for training predictive models for the trading bot.

This script orchestrates dataset loading, model training, evaluation and
serialisation by leveraging :mod:`trading_bot.predictive_model`.

It is intended for research and backtesting only; it must not be used to place
real market orders.  Models should always be validated on out-of-sample data
before deployment.

Example
-------
Run the script with a CSV dataset, specifying the target column and desired
model:

.. code-block:: bash

   python train_predictive_model.py datos.csv --target resultado \
       --model_type random_forest --cv_splits 5 --output modelo.pkl --verbose

This will train a Random Forest classifier using 5 folds for time-series
cross-validation, store the resulting artefacts in ``modelo.pkl`` and display a
summary of evaluation metrics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from trading_bot.predictive_model import (
    TrainingConfig,
    evaluate_model,
    load_model,
    save_model,
    train_model,
)

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the training utility."""

    parser = argparse.ArgumentParser(
        description="Train a predictive ML model for the trading bot",
    )
    parser.add_argument(
        "csv",
        type=str,
        help="Path to the CSV file containing the training dataset",
    )
    parser.add_argument(
        "--target",
        default="target",
        help="Name of the binary target column (default: %(default)s)",
    )
    parser.add_argument(
        "--model_type",
        choices=("logistic", "random_forest"),
        default="logistic",
        help="Type of model to train (default: %(default)s)",
    )
    parser.add_argument(
        "--cv_splits",
        type=int,
        default=3,
        help="Number of splits for time-series cross-validation (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="model.pkl",
        help="Destination file where the trained model will be saved",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate an existing model without retraining",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    """Initialise logging with a human-friendly format."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def load_dataset(csv_path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a dataset from ``csv_path`` and split it into features and target."""

    if not csv_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")

    LOGGER.debug("Loading dataset from %s", csv_path)
    frame = pd.read_csv(csv_path)

    if target not in frame.columns:
        raise ValueError(
            f"Target column '{target}' not present in dataset. Available columns: {list(frame.columns)}"
        )

    X = frame.drop(columns=[target])
    y = frame[target]
    return X, y


def display_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics returned by :func:`evaluate_model`."""

    lines = [
        "Evaluation summary:",
        f"  Accuracy        : {metrics['accuracy']:.4f}",
        f"  Precision       : {metrics['precision']:.4f}",
        f"  Recall          : {metrics['recall']:.4f}",
        f"  F1-score        : {metrics['f1']:.4f}",
        f"  ROC AUC         : {metrics['roc_auc']:.4f}",
    ]

    sharpe = metrics.get("sharpe_ratio")
    if sharpe is None:
        lines.append("  Sharpe ratio   : n/a")
    else:
        lines.append(f"  Sharpe ratio   : {sharpe:.4f}")

    lines.append("  Confusion matrix:")
    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, list):
        for row in confusion:
            lines.append("    " + "  ".join(f"{value:>6}" for value in row))

    roc_path = metrics.get("roc_curve_path")
    cm_path = metrics.get("confusion_matrix_path")
    if roc_path:
        lines.append(f"  ROC curve plot : {roc_path}")
    if cm_path:
        lines.append(f"  CM plot        : {cm_path}")

    for line in lines:
        print(line)


def run_evaluation(model_path: Path, csv_path: Path, target: str) -> int:
    """Load a model and dataset, then output evaluation metrics."""

    model = load_model(str(model_path))
    if model is None:
        LOGGER.error("Model file '%s' could not be found for evaluation", model_path)
        return 1

    X, y = load_dataset(csv_path, target)
    metrics = evaluate_model(model, X, y)
    display_metrics(metrics)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the command line interface."""

    args = parse_args(argv)
    configure_logging(args.verbose)

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.output).expanduser().resolve()

    try:
        if args.eval_only:
            LOGGER.info("Running in evaluation mode")
            return run_evaluation(model_path, csv_path, args.target)

        LOGGER.info("Training model using dataset at %s", csv_path)
        config = TrainingConfig(
            model_type=args.model_type,
            cv_splits=args.cv_splits,
        )

        model = train_model(str(csv_path), args.target, config)
        save_model(model, str(model_path))
        LOGGER.info("Model saved to %s", model_path)

        LOGGER.info("Evaluating trained model on the provided dataset")
        X, y = load_dataset(csv_path, args.target)
        metrics = evaluate_model(model, X, y)
        display_metrics(metrics)
        return 0

    except Exception as exc:
        if args.verbose:
            LOGGER.exception("An error occurred while executing the script")
        else:
            LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
