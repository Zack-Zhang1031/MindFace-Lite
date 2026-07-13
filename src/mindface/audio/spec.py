from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mindface.audio.features import extract_audio_features


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    fps: int = 25
    frame_ms: float = 40.0
    feature_dim: int = 70

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("FeatureSpec.fps must be positive")
        if self.frame_ms <= 0:
            raise ValueError("FeatureSpec.frame_ms must be positive")
        if self.feature_dim != 70:
            raise ValueError("MindFace-Lite currently supports feature_dim=70")

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return extract_audio_features(
            waveform,
            sample_rate,
            fps=self.fps,
            frame_ms=self.frame_ms,
            feature_dim=self.feature_dim,
        )

    def to_dict(self) -> dict[str, int | float]:
        return {"fps": self.fps, "frame_ms": self.frame_ms, "feature_dim": self.feature_dim}

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any] | None,
        model_config: dict[str, Any] | None = None,
    ) -> "FeatureSpec":
        values = values or {}
        model_config = model_config or {}
        return cls(
            fps=int(values.get("fps", 25)),
            frame_ms=float(values.get("frame_ms", 40.0)),
            feature_dim=int(values.get("feature_dim", model_config.get("input_dim", 70))),
        )

