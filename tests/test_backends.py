from __future__ import annotations

import numpy as np
import pytest
import torch

from mindface.artifacts.model_bundle import ModelBundle, save_model_bundle
from mindface.audio.spec import FeatureSpec
from mindface.backends import MouthPredictor, OnnxPredictor, PyTorchPredictor
from mindface.deploy.onnx_tools import export_checkpoint_to_onnx
from mindface.models.factory import build_model


def test_pytorch_and_onnx_backends_share_predictor_contract(tmp_path) -> None:
    model_config = {"input_dim": 70, "hidden_dim": 8, "output_dim": 3, "dropout": 0.0}
    model = build_model("mlp", model_config)
    checkpoint_path = tmp_path / "model.pt"
    save_model_bundle(
        ModelBundle(
            model_type="mlp",
            model_config=model_config,
            feature_spec=FeatureSpec(),
            model_state=model.state_dict(),
        ),
        checkpoint_path,
    )
    onnx_path = export_checkpoint_to_onnx(checkpoint_path, tmp_path / "model.onnx")
    features = np.random.default_rng(1).random((5, 70), dtype=np.float32)

    pytorch_backend = PyTorchPredictor.from_checkpoint(checkpoint_path)
    onnx_backend = OnnxPredictor(onnx_path, model_type="mlp", feature_dim=70)
    pytorch_output = pytorch_backend.predict(features)
    onnx_output = onnx_backend.predict(features)

    assert isinstance(pytorch_backend, MouthPredictor)
    assert isinstance(onnx_backend, MouthPredictor)
    assert pytorch_output.shape == (5, 3)
    assert np.allclose(pytorch_output, onnx_output, atol=1e-5)
    assert pytorch_backend.metadata()["backend"] == "pytorch"
    assert onnx_backend.metadata()["backend"] == "onnxruntime"


def test_backend_rejects_wrong_feature_shape(tmp_path) -> None:
    model_config = {"input_dim": 70, "hidden_dim": 8, "output_dim": 3, "dropout": 0.0}
    model = build_model("mlp", model_config)
    checkpoint_path = tmp_path / "model.pt"
    save_model_bundle(
        ModelBundle(
            model_type="mlp",
            model_config=model_config,
            feature_spec=FeatureSpec(),
            model_state=model.state_dict(),
        ),
        checkpoint_path,
    )
    backend = PyTorchPredictor.from_checkpoint(checkpoint_path)

    with pytest.raises(ValueError, match="feature_dim=70"):
        backend.predict(np.zeros((3, 12), dtype=np.float32))

