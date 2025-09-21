"""Predictive modeling utilities for the trading bot package.

This module provides tools to train, evaluate, persist and use machine learning
classifiers that estimate the probability of success for a trade.  The models
are meant for research and backtesting workflows; they do not guarantee
profitability and **must not** be used to send real market orders without proper
supervision.  Typical integration points include loading a trained model when
the bot starts (``load_model(config.MODEL_PATH)``) and combining its
``predict_proba`` output with existing heuristic probabilities to obtain a
blended ``prob_success`` score.

Example usage
-------------
>>> from trading_bot import predictive_model
>>> cfg = predictive_model.TrainingConfig(model_type="logistic")
>>> model = predictive_model.train_model("dataset.csv", "is_profitable", cfg)
>>> metrics = predictive_model.evaluate_model(model, X_test, y_test)
>>> predictive_model.save_model(model, "models/latest.pkl")
>>> restored = predictive_model.load_model("models/latest.pkl")
>>> restored.predict_proba(X_live)[:, 1]

The module automatically handles preprocessing (scaling numeric features and
encoding categorical ones), performs hyper-parameter search with
``GridSearchCV`` and stores evaluation artefacts in the ``reports/`` directory.

Integration notes
-----------------
* ``strategy.decidir_entrada`` prepares a frame with columns
  ``["risk_reward", "orig_prob", "side"]``.  The training CSV must expose the
  same column names (plus the binary target column) so that live predictions
  remain compatible.
* When updating ``prob_success``, average the heuristic score with
  ``predict_proba`` results using a weighting scheme appropriate for your risk
  appetite.
* Always validate the model on hold-out data and monitor drift before applying
  it to live decisions.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union, cast

import matplotlib

# Ensure plotting works in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

LOGGER = logging.getLogger(__name__)


def _manifest_path(path: Union[str, Path]) -> Path:
    destination = Path(path)
    return destination.with_name(destination.stem + ".manifest.json")


@dataclass
class TrainingConfig:
    """Configuration options that control model training.

    Parameters
    ----------
    model_type:
        Identifier of the estimator to train. Supported values are ``"logistic"``
        and ``"random_forest"``.
    hyperparams:
        Optional grid of hyper-parameters to explore. When ``None`` sensible
        defaults are used for the selected model. Keys may omit the
        ``classifier__`` prefix; it will be added automatically.
    cv_splits:
        Number of splits for ``TimeSeriesSplit`` during cross-validation.
    use_polynomial_features:
        If ``True`` and ``model_type`` is ``"logistic"``, second-degree
        polynomial features are generated for numeric columns in order to
        capture mild non-linearities.
    random_state:
        Seed passed to underlying estimators to make experiments reproducible.
    n_jobs:
        Degree of parallelism for ``GridSearchCV``. ``None`` maps to scikit-learn
        defaults (use all available cores).
    """

    model_type: str = "logistic"
    hyperparams: Optional[Dict[str, Iterable[Any]]] = None
    cv_splits: int = 5
    use_polynomial_features: bool = False
    random_state: int = 42
    n_jobs: Optional[int] = None
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"
    calibration_cv: int = 3
    _validated_model_type: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        supported = {"logistic", "random_forest"}
        if self.model_type not in supported:
            raise ValueError(f"Unsupported model_type '{self.model_type}'. "
                             f"Choose from {sorted(supported)}")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2 for TimeSeriesSplit")
        if self.calibration_method not in {"isotonic", "sigmoid"}:
            raise ValueError("calibration_method must be 'isotonic' or 'sigmoid'")
        if self.calibration_cv < 1:
            raise ValueError("calibration_cv must be at least 1")
        self._validated_model_type = self.model_type


def train_model(
    csv_path: str,
    target: str,
    config: Optional[TrainingConfig] = None,
) -> BaseEstimator:
    """Train a predictive model using data stored in a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the dataset in CSV format. The CSV must contain the target
        column and the feature columns expected at inference time (for example
        ``risk_reward``, ``orig_prob`` and ``side``).
    target:
        Name of the binary target column (1 for profitable trades, 0 otherwise).
    config:
        Optional :class:`TrainingConfig` that controls preprocessing and
        hyper-parameter search. When omitted a default configuration for
        logistic regression is used.

    Returns
    -------
    BaseEstimator
        The fitted scikit-learn pipeline containing preprocessing and the best
        estimator found by the grid search.

    Raises
    ------
    ValueError
        If the CSV is missing the target column or contains insufficient data.
    """

    cfg = config or TrainingConfig()
    data_path = Path(csv_path)
    if not data_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")

    LOGGER.info("Loading dataset from %s", data_path)
    frame = pd.read_csv(data_path)

    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    if len(frame) < cfg.cv_splits + 1:
        raise ValueError(
            "Dataset has too few rows for the requested number of CV splits"
        )

    y = frame[target]
    X = frame.drop(columns=[target])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    LOGGER.debug("Numeric features: %s", numeric_features)
    LOGGER.debug("Categorical features: %s", categorical_features)

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if cfg.use_polynomial_features and cfg.model_type == "logistic" and numeric_features:
        numeric_steps.append(
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        )
    numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    estimator: BaseEstimator
    param_grid: Dict[str, Iterable[Any]]
    if cfg.model_type == "logistic":
        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=cfg.random_state,
        )
        param_grid = {
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        }
    else:
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_split": [2, 5],
        }

    if cfg.hyperparams:
        param_grid = {}
        for key, value in cfg.hyperparams.items():
            prefixed_key = key if key.startswith("classifier__") else f"classifier__{key}"
            param_grid[prefixed_key] = value

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    LOGGER.info(
        "Starting grid search for model_type=%s with %d CV splits",
        cfg.model_type,
        cfg.cv_splits,
    )
    cv = TimeSeriesSplit(n_splits=cfg.cv_splits)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=cfg.n_jobs,
        refit=True,
        verbose=0,
    )
    grid.fit(X, y)

    LOGGER.info(
        "Best params: %s (ROC AUC: %.4f)",
        grid.best_params_,
        grid.best_score_,
    )

    manifest = {
        "features": list(X.columns),
        "numeric": list(numeric_features),
        "categorical": list(categorical_features),
        "target": target,
        "model_type": cfg.model_type,
        "calibrated": cfg.calibrate_probabilities,
        "created_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }

    best_pipeline = grid.best_estimator_

    if cfg.calibrate_probabilities:
        preprocessor = clone(best_pipeline.named_steps["preprocessor"])
        classifier = clone(best_pipeline.named_steps["classifier"])
        calibrator = CalibratedClassifierCV(
            estimator=classifier,
            method=cfg.calibration_method,
            cv=cfg.calibration_cv,
        )
        calibrated_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", calibrator),
            ]
        )
        calibrated_pipeline.fit(X, y)
        calibrated_pipeline.feature_manifest_ = manifest
        return calibrated_pipeline

    best_pipeline.feature_manifest_ = manifest
    return best_pipeline


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    returns: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Evaluate a fitted model on a labelled dataset.

    Parameters
    ----------
    model:
        Trained scikit-learn estimator supporting ``predict`` and
        ``predict_proba``.
    X:
        Feature matrix containing the same columns used during training.
    y:
        Ground truth labels (1 for successful trades, 0 otherwise).
    returns:
        Optional sequence representing the realised return of each operation.
        When provided the function computes a simulated annualised Sharpe ratio.

    Returns
    -------
    Dict[str, Any]
        Dictionary with computed metrics and paths to generated figures.
    """

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("Model must implement predict and predict_proba methods")

    predictions = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)

    conf_matrix = confusion_matrix(y, predictions)

    sharpe_ratio: Optional[float] = None
    if returns is not None:
        returns_array = np.asarray(list(returns), dtype=float)
        if returns_array.size != len(y):
            raise ValueError("Returns vector must match the number of samples")
        volatility = returns_array.std(ddof=1)
        if np.isclose(volatility, 0.0):
            sharpe_ratio = None
        else:
            sharpe_ratio = (returns_array.mean() / volatility) * np.sqrt(252)

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    roc_path = reports_dir / "roc_curve.png"
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)

    cm_path = reports_dir / "confusion_matrix.png"
    fig_cm, ax_cm = plt.subplots()
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    display.plot(ax=ax_cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    LOGGER.info("Evaluation complete. Accuracy=%.3f AUC=%.3f", accuracy, roc_auc)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),
        "sharpe_ratio": sharpe_ratio,
        "roc_curve_path": str(roc_path),
        "confusion_matrix_path": str(cm_path),
    }


def save_model(model: BaseEstimator, path: str) -> None:
    """Persist a fitted model to disk using :mod:`pickle`.

    Parameters
    ----------
    model:
        Trained estimator or pipeline to be serialised.
    path:
        Destination path (``.pkl``) where the model will be stored. Parent
        directories are created automatically.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump(model, file)
    LOGGER.info("Model saved to %s", destination)

    manifest = getattr(model, "feature_manifest_", None)
    if manifest:
        manifest_path = _manifest_path(destination)
        with manifest_path.open("w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        LOGGER.info("Feature manifest saved to %s", manifest_path)


def load_model(path: str) -> Optional[BaseEstimator]:
    """Load a previously saved model from disk.

    Parameters
    ----------
    path:
        Location of the pickled model.

    Returns
    -------
    Optional[BaseEstimator]
        The deserialised estimator, or ``None`` if the file does not exist.
    """

    model_path = Path(path)
    if not model_path.exists():
        LOGGER.warning("Requested model path does not exist: %s", model_path)
        return None

    with model_path.open("rb") as file:
        model = pickle.load(file)
    manifest_path = _manifest_path(model_path)
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            setattr(model, "feature_manifest_", manifest)
            LOGGER.info(
                "Loaded model and manifest from %s and %s",
                model_path,
                manifest_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed loading feature manifest %s: %s", manifest_path, exc)
    else:
        LOGGER.info("Model loaded from %s", model_path)
    return cast(BaseEstimator, model)


def ensure_feature_schema(
    model: BaseEstimator, frame: pd.DataFrame
) -> pd.DataFrame:
    """Validate that ``frame`` matches the model feature schema."""

    manifest = getattr(model, "feature_manifest_", None)
    if not manifest:
        return frame

    expected = manifest.get("features", [])
    missing = [col for col in expected if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    ordered = frame[expected].copy()

    numeric = set(manifest.get("numeric", []))
    categorical = set(manifest.get("categorical", []))

    for col in numeric:
        if col not in ordered.columns:
            continue
        if not is_numeric_dtype(ordered[col]):
            raise TypeError(f"Feature '{col}' must be numeric")

    for col in categorical:
        if col not in ordered.columns:
            continue
        if not (
            is_categorical_dtype(ordered[col])
            or ordered[col].dtype.kind in {"O", "U"}
        ):
            ordered[col] = ordered[col].astype(str)

    return ordered


def predict_proba(
    model: BaseEstimator,
    X: Union[pd.DataFrame, Dict[str, Any], Sequence[Dict[str, Any]]],
) -> np.ndarray:
    """Predict success probabilities for the positive class.

    Parameters
    ----------
    model:
        Fitted estimator that exposes ``predict_proba``.
    X:
        Feature matrix or structure compatible with the training columns. It can
        be a pandas ``DataFrame``, a dictionary representing a single sample or a
        sequence of dictionaries.

    Returns
    -------
    numpy.ndarray
        Array with the predicted probability of the positive class for each
        sample.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba")

    prepared_X: pd.DataFrame
    if isinstance(X, pd.DataFrame):
        prepared_X = X
    elif isinstance(X, dict):
        prepared_X = pd.DataFrame([X])
    elif isinstance(X, Sequence):
        # Sequence of dictionaries or mappings.
        prepared_X = pd.DataFrame(list(X))
    else:
        raise TypeError("X must be a DataFrame, dict or sequence of dicts")

    prepared_X = ensure_feature_schema(model, prepared_X)

    try:
        probabilities = model.predict_proba(prepared_X)[:, 1]
    except NotFittedError as exc:
        raise ValueError("Model instance is not fitted") from exc

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predicted probabilities must lie within [0, 1]")

    return probabilities


__all__ = [
    "TrainingConfig",
    "train_model",
    "evaluate_model",
    "save_model",
    "load_model",
    "ensure_feature_schema",
    "predict_proba",
]

