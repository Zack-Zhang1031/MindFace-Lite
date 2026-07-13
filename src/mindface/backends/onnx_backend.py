from __future__ import annotations

from pathlib import Path

import numpy as np

from mindface.backends.base import validate_features
from mindface.models.factory import is_sequence_model
from mindface.utils.config import resolve_path


class OnnxPredictor:
    def __init__(
        self,
        onnx_path: str | Path,
        model_type: str,
        feature_dim: int = 70,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed") from exc
        self.path = resolve_path(onnx_path)
        if not self.path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.path}")
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.path), providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = validate_features(features, self.feature_dim)
        input_values = values[None, :, :] if is_sequence_model(self.model_type) else values
        output = self.session.run(None, {self.input_name: input_values})[0]
        if is_sequence_model(self.model_type):
            output = output[0]
        return np.asarray(output, dtype=np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "backend": "onnxruntime",
            "model_type": self.model_type,
            "feature_dim": self.feature_dim,
            "path": str(self.path),
            "providers": self.session.get_providers(),
        }

