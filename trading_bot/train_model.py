import argparse
import pandas as pd
from . import optimizer


def main():
    parser = argparse.ArgumentParser(
        description="Train ML model for trading bot"
    )
    parser.add_argument("csv", help="CSV file with training data")
    parser.add_argument(
        "--target",
        default="target",
        help="Target column name",
    )
    parser.add_argument(
        "--model",
        default="model.pkl",
        help="Output model path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    model = optimizer.optimize_model(df, args.target)
    optimizer.save_model(model, args.model)
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
