from __future__ import annotations

import numpy as np
import torch

from mindface.artifacts.model_bundle import ModelBundle, load_model_bundle, save_model_bundle
from mindface.audio.spec import FeatureSpec
from mindface.models.factory import build_model


def test_feature_spec_is_the_single_audio_feature_contract() -> None:
    spec = FeatureSpec(fps=20, frame_ms=50.0, feature_dim=70)
    waveform = np.zeros(1600, dtype=np.float32)

    features = spec.extract(waveform, sample_rate=16000)

    assert features.shape == (2, 70)
    assert spec.to_dict() == {"fps": 20, "frame_ms": 50.0, "feature_dim": 70}


def test_model_bundle_round_trip_preserves_training_state(tmp_path) -> None:
    model_config = {"input_dim": 70, "hidden_dim": 8, "output_dim": 3, "dropout": 0.0}
    model = build_model("mlp", model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bundle = ModelBundle(
        model_type="mlp",
        model_config=model_config,
        feature_spec=FeatureSpec(),
        target_names=("mouth_open", "mouth_width", "lip_round"),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        epoch=2,
        best_val_loss=0.125,
        train_config={"epochs": 3},
        metadata={"source": "unit-test"},
    )
    path = tmp_path / "bundle.pt"

    save_model_bundle(bundle, path)
    loaded = load_model_bundle(path)

    assert loaded.schema_version == 1
    assert loaded.feature_spec == FeatureSpec()
    assert loaded.target_names == ("mouth_open", "mouth_width", "lip_round")
    assert loaded.epoch == 2
    assert loaded.optimizer_state is not None
    assert loaded.metadata["source"] == "unit-test"


def test_legacy_checkpoint_is_migrated_to_model_bundle(tmp_path) -> None:
    model_config = {"input_dim": 70, "hidden_dim": 8, "output_dim": 3, "dropout": 0.0}
    model = build_model("mlp", model_config)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_type": "mlp",
            "model_config": model_config,
            "model_state": model.state_dict(),
            "best_val_loss": 0.5,
            "feature_config": {"fps": 25, "frame_ms": 40},
            "train_config": {"epochs": 1},
        },
        path,
    )

    loaded = load_model_bundle(path)

    assert loaded.schema_version == 1
    assert loaded.epoch == 0
    assert loaded.feature_spec.feature_dim == 70
    assert loaded.optimizer_state is None

