"""Tests for the predictive modeling helpers."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading_bot import predictive_model
from sklearn.calibration import CalibratedClassifierCV


def _make_dataset() -> pd.DataFrame:
    """Create a minimal but non-trivial dataset for tests."""

    rng = np.random.default_rng(42)
    rows = 60
    data = {
        "risk_reward": rng.normal(loc=1.5, scale=0.3, size=rows),
        "orig_prob": rng.uniform(0.4, 0.7, size=rows),
        "side": rng.choice(["long", "short"], size=rows),
        "momentum": rng.normal(loc=0.0, scale=1.0, size=rows),
    }
    frame = pd.DataFrame(data)
    logits = 2 * frame["orig_prob"] + 0.5 * frame["momentum"]
    frame["is_profitable"] = (logits + rng.normal(0, 0.5, size=rows) > 1.1).astype(int)
    return frame


def test_train_model_missing_target(tmp_path: Path) -> None:
    dataset = _make_dataset().drop(columns=["is_profitable"])
    csv_path = tmp_path / "train.csv"
    dataset.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Target column 'is_profitable' not found"):
        predictive_model.train_model(str(csv_path), "is_profitable")


def test_predict_proba_range_and_order(tmp_path: Path) -> None:
    dataset = _make_dataset()
    csv_path = tmp_path / "train.csv"
    dataset.to_csv(csv_path, index=False)

    model = predictive_model.train_model(str(csv_path), "is_profitable")

    # Use two specific samples to assert ordering is maintained.
    samples = dataset.iloc[:2, :-1]
    probs = predictive_model.predict_proba(model, samples)

    assert probs.shape == (2,)
    assert np.all((0.0 <= probs) & (probs <= 1.0))

    # Recompute by passing a dict and verify equivalence for single sample.
    dict_prob = predictive_model.predict_proba(model, samples.iloc[0].to_dict())
    assert dict_prob.shape == (1,)
    assert pytest.approx(dict_prob[0]) == probs[0]


def test_model_serialisation_roundtrip(tmp_path: Path) -> None:
    dataset = _make_dataset()
    csv_path = tmp_path / "train.csv"
    dataset.to_csv(csv_path, index=False)

    config = predictive_model.TrainingConfig(model_type="random_forest", cv_splits=3)
    model = predictive_model.train_model(str(csv_path), "is_profitable", config=config)

    X_eval = dataset.iloc[5:15, :-1]
    expected = predictive_model.predict_proba(model, X_eval)

    model_path = tmp_path / "model.pkl"
    predictive_model.save_model(model, str(model_path))

    # Load manually via module helper and check predictions match.
    loaded = predictive_model.load_model(str(model_path))
    assert loaded is not None
    restored_probs = predictive_model.predict_proba(loaded, X_eval)
    np.testing.assert_allclose(expected, restored_probs)

    # Ensure the pickled payload is compatible with vanilla pickle.load as well.
    with model_path.open("rb") as file:
        direct = pickle.load(file)
    direct_probs = predictive_model.predict_proba(direct, X_eval)
    np.testing.assert_allclose(expected, direct_probs)


def test_feature_manifest_saved_and_validated(tmp_path: Path) -> None:
    dataset = _make_dataset()
    csv_path = tmp_path / "train.csv"
    dataset.to_csv(csv_path, index=False)

    model = predictive_model.train_model(str(csv_path), "is_profitable")
    manifest = getattr(model, "feature_manifest_", None)
    assert manifest is not None
    assert set(manifest["features"]) == set(dataset.columns[:-1])

    model_path = tmp_path / "model.pkl"
    predictive_model.save_model(model, str(model_path))
    manifest_path = model_path.with_name(model_path.stem + ".manifest.json")
    assert manifest_path.exists()

    loaded = predictive_model.load_model(str(model_path))
    assert loaded is not None

    valid = dataset.iloc[:1, :-1].copy()
    prepared = predictive_model.ensure_feature_schema(loaded, valid)
    assert list(prepared.columns) == manifest["features"]

    missing = valid.drop(columns=[manifest["features"][0]])
    with pytest.raises(ValueError, match="Missing required feature"):
        predictive_model.ensure_feature_schema(loaded, missing)

    wrong_type = valid.copy()
    wrong_type[manifest["numeric"][0]] = "oops"
    with pytest.raises(TypeError, match="must be numeric"):
        predictive_model.ensure_feature_schema(loaded, wrong_type)


def test_training_uses_calibrated_classifier(tmp_path: Path) -> None:
    dataset = _make_dataset()
    csv_path = tmp_path / "train.csv"
    dataset.to_csv(csv_path, index=False)

    config = predictive_model.TrainingConfig(
        model_type="logistic",
        calibrate_probabilities=True,
        calibration_cv=3,
    )
    model = predictive_model.train_model(str(csv_path), "is_profitable", config=config)
    classifier = model.named_steps["classifier"]
    assert isinstance(classifier, CalibratedClassifierCV)

