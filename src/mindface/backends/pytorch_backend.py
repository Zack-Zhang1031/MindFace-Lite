from __future__ import annotations

from pathlib import Path

import numpy as np

from mindface.artifacts.model_bundle import ModelBundle, load_model_bundle
from mindface.backends.base import validate_features
from mindface.inference import predict_from_features
from mindface.models.factory import build_model


class PyTorchPredictor:
    def __init__(self, bundle: ModelBundle, device: str = "cpu") -> None:
        self.bundle = bundle
        self.device = device
        self.model = build_model(bundle.model_type, bundle.model_config)
        self.model.load_state_dict(bundle.model_state)
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu") -> "PyTorchPredictor":
        return cls(load_model_bundle(checkpoint_path, map_location=device), device=device)

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = validate_features(features, self.bundle.feature_spec.feature_dim)
        return predict_from_features(self.model, self.bundle.model_type, values, device=self.device)

    def metadata(self) -> dict[str, object]:
        return {
            "backend": "pytorch",
            "model_type": self.bundle.model_type,
            "feature_spec": self.bundle.feature_spec.to_dict(),
            "device": self.device,
        }

