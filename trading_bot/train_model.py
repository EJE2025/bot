"""Minimal CLI wrapper around :mod:`trading_bot.predictive_model`.

This script is retained for backwards compatibility with earlier automation
that depended on ``trading_bot.train_model``. Internally it now delegates to the
more feature-rich :mod:`trading_bot.predictive_model` utilities to avoid
maintaining two independent training paths.
"""

from __future__ import annotations

import argparse

from .predictive_model import TrainingConfig, evaluate_model, save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a machine-learning model for the trading bot",
    )
    parser.add_argument("csv", help="CSV file with training data")
    parser.add_argument(
        "--target",
        default="target",
        help="Target column name (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="model.pkl",
        help="Output model path (default: %(default)s)",
    )
    parser.add_argument(
        "--model-type",
        choices=("logistic", "random_forest"),
        default="random_forest",
        help="Underlying estimator to use (default: %(default)s)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of time-series CV splits (default: %(default)s)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable probability calibration after training",
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Evaluate the trained model on the training set and print metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainingConfig(
        model_type=args.model_type,
        cv_splits=args.cv_splits,
        calibrate_probabilities=not args.no_calibration,
    )

    model = train_model(args.csv, args.target, cfg)
    save_model(model, args.output)
    print(f"Model saved to {args.output}")

    if args.print_metrics:
        # Lazily import pandas to keep the CLI lightweight when metrics are not required.
        import pandas as pd

        frame = pd.read_csv(args.csv)
        X = frame.drop(columns=[args.target])
        y = frame[args.target]
        metrics = evaluate_model(model, X, y)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
