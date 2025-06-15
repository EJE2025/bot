import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

logger = logging.getLogger(__name__)


def optimize_model(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError("Target column not in dataframe")
    features = df.drop(columns=[target])
    labels = df[target]
    model = RandomForestClassifier(n_estimators=50)
    model.fit(features, labels)
    logger.info("Model optimized with %d samples", len(df))
    return model


def save_model(model, path: str = "model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = "model.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


