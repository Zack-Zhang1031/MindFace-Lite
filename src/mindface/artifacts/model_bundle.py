from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from mindface.audio.spec import FeatureSpec
from mindface.utils.config import resolve_path


MODEL_BUNDLE_SCHEMA_VERSION = 1
DEFAULT_TARGET_NAMES = ("mouth_open", "mouth_width", "lip_round")


@dataclass(slots=True)
class ModelBundle:
    model_type: str
    model_config: dict[str, Any]
    feature_spec: FeatureSpec
    model_state: dict[str, Any]
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES
    optimizer_state: dict[str, Any] | None = None
    epoch: int = 0
    best_val_loss: float = float("inf")
    train_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = MODEL_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported model bundle schema_version={self.schema_version}")
        if not self.model_type:
            raise ValueError("model_type must not be empty")
        if not self.target_names:
            raise ValueError("target_names must not be empty")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "artifact_type": "mindface_model_bundle",
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "model_config": self.model_config,
            "feature_spec": self.feature_spec.to_dict(),
            "feature_config": self.feature_spec.to_dict(),
            "target_names": list(self.target_names),
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "train_config": self.train_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> "ModelBundle":
        if "model_type" not in checkpoint or "model_config" not in checkpoint or "model_state" not in checkpoint:
            raise ValueError("Checkpoint is missing model_type, model_config, or model_state")
        model_config = dict(checkpoint["model_config"])
        feature_values = checkpoint.get("feature_spec", checkpoint.get("feature_config", {}))
        return cls(
            schema_version=int(checkpoint.get("schema_version", MODEL_BUNDLE_SCHEMA_VERSION)),
            model_type=str(checkpoint["model_type"]),
            model_config=model_config,
            feature_spec=FeatureSpec.from_mapping(feature_values, model_config),
            target_names=tuple(checkpoint.get("target_names", DEFAULT_TARGET_NAMES)),
            model_state=dict(checkpoint["model_state"]),
            optimizer_state=checkpoint.get("optimizer_state"),
            epoch=int(checkpoint.get("epoch", 0)),
            best_val_loss=float(checkpoint.get("best_val_loss", float("inf"))),
            train_config=dict(checkpoint.get("train_config", {})),
            metadata=dict(checkpoint.get("metadata", {})),
        )


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.to_checkpoint(), resolved)
    return resolved


def load_model_bundle(path: str | Path, map_location: str | torch.device = "cpu") -> ModelBundle:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    checkpoint = torch.load(resolved, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a mapping: {resolved}")
    return ModelBundle.from_checkpoint(checkpoint)

