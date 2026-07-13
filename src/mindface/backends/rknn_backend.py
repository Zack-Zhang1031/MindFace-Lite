from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mindface.backends.base import validate_features
from mindface.utils.config import resolve_path


class RknnPredictor:
    def __init__(self, rknn_path: str | Path, feature_dim: int = 70, runtime_target: str | None = None) -> None:
        self.path = resolve_path(rknn_path)
        self.feature_dim = feature_dim
        self.runtime_target = runtime_target
        self._runtime: Any | None = None

    def _get_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if not self.path.exists():
            raise FileNotFoundError(f"RKNN file not found: {self.path}")
        try:
            from rknn.api import RKNN
        except ImportError as exc:
            raise RuntimeError(f"rknn-toolkit2 is not installed: {exc}") from exc
        runtime = RKNN(verbose=False)
        if runtime.load_rknn(str(self.path)) != 0:
            runtime.release()
            raise RuntimeError(f"rknn.load_rknn failed: {self.path}")
        init_code = runtime.init_runtime(target=self.runtime_target) if self.runtime_target else runtime.init_runtime()
        if init_code != 0:
            runtime.release()
            raise RuntimeError(f"rknn.init_runtime failed with code {init_code}")
        self._runtime = runtime
        return runtime

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = validate_features(features, self.feature_dim)
        outputs = self._get_runtime().inference(inputs=[values])
        return np.asarray(outputs[0], dtype=np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "backend": "rknn",
            "feature_dim": self.feature_dim,
            "path": str(self.path),
            "runtime_target": self.runtime_target or "local",
        }

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.release()
            self._runtime = None

