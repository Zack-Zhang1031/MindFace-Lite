from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mindface.models.factory import build_model, is_sequence_model
from mindface.utils.config import resolve_path


def export_checkpoint_to_onnx(checkpoint_path: str | Path, output_path: str | Path, opset: int = 12) -> Path:
    checkpoint_path = resolve_path(checkpoint_path)
    output_path = resolve_path(output_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_type = checkpoint["model_type"]
    model = build_model(model_type, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_dim = int(checkpoint["model_config"].get("input_dim", 70))
    if is_sequence_model(model_type):
        dummy = torch.randn(1, 16, input_dim, dtype=torch.float32)
        dynamic_axes = {"audio_features": {0: "batch", 1: "frames"}, "mouth_params": {0: "batch", 1: "frames"}}
    else:
        dummy = torch.randn(16, input_dim, dtype=torch.float32)
        dynamic_axes = {"audio_features": {0: "frames"}, "mouth_params": {0: "frames"}}

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["audio_features"],
        output_names=["mouth_params"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    return output_path


def run_onnx_inference(onnx_path: str | Path, features: np.ndarray, model_type: str = "mlp") -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime") from exc

    onnx_path = resolve_path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    features = np.asarray(features, dtype=np.float32)
    if is_sequence_model(model_type):
        input_data = features[None, :, :]
        output = session.run(None, {"audio_features": input_data})[0][0]
    else:
        output = session.run(None, {"audio_features": features})[0]
    return np.asarray(output, dtype=np.float32)
