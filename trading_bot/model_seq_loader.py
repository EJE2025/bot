"""Sequential model loading and inference helpers.

This module centralises the logic required to load sequence-based models trained
with either TensorFlow/Keras or PyTorch. It ensures models are moved to CPU,
put into evaluation mode and called in a consistent way from the strategy.

Expected training/inference contract
------------------------------------
* Inputs must have shape ``(batch, seq_len, features)``.
* Features are ordered as ``[close, high, low, volume]`` and should be scaled
  with feature-wise z-score normalisation over the sequence window used for
  inference (mean/std computed per feature on the window, using a small epsilon
  to avoid division by zero). Training pipelines should mirror this exact
  normalisation so that live data matches the distribution seen during training.
* Model outputs are assumed to be probabilities in ``[0, 1]`` (e.g. sigmoid on
  a single logit or softmax with the positive class in index 1). Any output is
  clipped to ``[0, 1]`` defensively before being returned.
"""

from __future__ import annotations

import logging
import importlib
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class SequenceModelError(RuntimeError):
    """Raised when a sequential model cannot be loaded or used."""


def _guess_backend(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return "torch"
    if suffix in {".h5", ".hdf5", ".keras"}:
        return "keras"
    # Default to Keras as it can load SavedModel directories too.
    return "keras"


def load_sequence_model(path: str) -> Optional[Any]:
    """Load a sequential model from ``path`` using the inferred backend."""

    model_path = Path(path)
    if not model_path.exists():
        LOGGER.warning("Sequential model not found at %s", path)
        return None

    backend = _guess_backend(model_path)

    if backend == "torch":
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            raise SequenceModelError("PyTorch must be installed to load .pt/.pth models")
        torch = importlib.import_module("torch")
        try:
            model = torch.load(model_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive import/load path
            raise SequenceModelError(f"Failed to load sequential model: {exc}") from exc
        if hasattr(model, "eval"):
            model.eval()
    else:
        keras_spec = importlib.util.find_spec("tensorflow.keras")
        if keras_spec is None:
            raise SequenceModelError("TensorFlow/Keras must be installed to load .h5/.keras models")
        keras = importlib.import_module("tensorflow.keras")
        try:
            model = keras.models.load_model(model_path, compile=False)
        except Exception as exc:  # pragma: no cover - defensive import/load path
            raise SequenceModelError(f"Failed to load sequential model: {exc}") from exc

    LOGGER.info("Sequential model loaded from %s using backend=%s", path, backend)
    return model


def predict_sequence(model: Any, sequence: np.ndarray) -> np.ndarray:
    """Run inference on ``sequence`` returning probabilities as ndarray."""

    if sequence.ndim != 3:
        raise ValueError("Sequence input must have shape (batch, seq_len, features)")

    torch_spec = importlib.util.find_spec("torch")
    torch_module = importlib.import_module("torch") if torch_spec else None

    if torch_module is not None and isinstance(model, torch_module.nn.Module):
        model.eval()
        with torch_module.no_grad():
            tensor = torch_module.tensor(sequence, dtype=torch_module.float32)
            output = model(tensor)
            probs = output.detach().cpu().numpy()
    else:
        if not hasattr(model, "predict"):
            raise SequenceModelError("Model does not expose a predict method")
        probs = np.asarray(model.predict(sequence))

    probs = np.squeeze(probs)
    return np.clip(probs, 0.0, 1.0)


__all__ = ["load_sequence_model", "predict_sequence", "SequenceModelError"]
