from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MouthPredictor(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict mouth parameters from a [frames, feature_dim] array."""

    def metadata(self) -> dict[str, object]:
        """Return backend identity and model input metadata."""


def validate_features(features: np.ndarray, feature_dim: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != feature_dim:
        raise ValueError(f"Expected features with shape [frames, feature_dim={feature_dim}], got {values.shape}")
    return values

