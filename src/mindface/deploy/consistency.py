from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mindface.artifacts.model_bundle import load_model_bundle
from mindface.audio.features import read_wav_mono
from mindface.audio.spec import FeatureSpec
from mindface.backends import OnnxPredictor, PyTorchPredictor, RknnPredictor
from mindface.deploy.onnx_tools import export_checkpoint_to_onnx
from mindface.utils.config import resolve_path


def _summary_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    length = min(len(reference), len(candidate))
    if length <= 0:
        return {"mae": float("nan"), "max_abs_error": float("nan"), "rmse": float("nan")}
    diff = np.asarray(reference[:length] - candidate[:length], dtype=np.float32)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _timed_call(fn, *args, **kwargs):
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, float((time.perf_counter() - started) * 1000.0)


def _run_rknn_optional(
    rknn_path: Path,
    features: np.ndarray,
    runtime_target: str | None,
    feature_dim: int,
) -> dict[str, Any]:
    predictor = RknnPredictor(rknn_path, feature_dim=feature_dim, runtime_target=runtime_target)
    try:
        output, latency_ms = _timed_call(predictor.predict, features)
        return {
            "available": True,
            "latency_ms": latency_ms,
            "output": output,
            "runtime_target": runtime_target or "local",
        }
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return {"available": False, "reason": str(exc)}
    finally:
        predictor.close()


def compare_backends_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = resolve_path(cfg["checkpoint"]["path"])
    onnx_path = resolve_path(cfg["onnx"]["path"])
    if bool(cfg["onnx"].get("auto_export", True)) and not onnx_path.exists():
        export_checkpoint_to_onnx(checkpoint_path, onnx_path, opset=int(cfg["onnx"].get("opset", 12)))

    bundle = load_model_bundle(checkpoint_path)
    requested_spec = FeatureSpec(
        fps=int(cfg["audio"]["fps"]),
        frame_ms=float(cfg["audio"]["frame_ms"]),
        feature_dim=bundle.feature_spec.feature_dim,
    )
    if requested_spec != bundle.feature_spec:
        raise ValueError(
            f"Consistency config FeatureSpec {requested_spec} does not match checkpoint {bundle.feature_spec}"
        )
    sample_rate, waveform = read_wav_mono(cfg["audio"]["input_path"])
    features = bundle.feature_spec.extract(waveform, sample_rate)
    max_frames = int(cfg["compare"].get("max_frames", 64))
    features = features[:max_frames].astype(np.float32)

    torch_predictor = PyTorchPredictor.from_checkpoint(
        checkpoint_path,
        device=str(cfg["pytorch"].get("device", "cpu")),
    )
    onnx_predictor = OnnxPredictor(
        onnx_path,
        model_type=bundle.model_type,
        feature_dim=bundle.feature_spec.feature_dim,
    )
    torch_output, torch_latency_ms = _timed_call(torch_predictor.predict, features)
    onnx_output, onnx_latency_ms = _timed_call(onnx_predictor.predict, features)

    report: dict[str, Any] = {
        "audio_path": str(resolve_path(cfg["audio"]["input_path"])),
        "frames": int(len(features)),
        "feature_shape": list(features.shape),
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "model_type": bundle.model_type,
        "feature_spec": bundle.feature_spec.to_dict(),
        "backends": {
            "pytorch": {
                "available": True,
                "latency_ms": torch_latency_ms,
                "output_shape": list(torch_output.shape),
            },
            "onnxruntime": {
                "available": True,
                "latency_ms": onnx_latency_ms,
                "output_shape": list(onnx_output.shape),
                "error_vs_pytorch": _summary_error(torch_output, onnx_output),
            },
        },
    }

    rknn_cfg = cfg.get("rknn", {})
    if bool(rknn_cfg.get("enabled", True)):
        rknn_path = resolve_path(rknn_cfg.get("path", "outputs/models/mlp_mouth.rk3588.rknn"))
        rknn_frames = int(rknn_cfg.get("frames", min(16, len(features))))
        rknn_input = features[:rknn_frames]
        rknn_result = _run_rknn_optional(
            rknn_path,
            rknn_input,
            rknn_cfg.get("runtime_target"),
            bundle.feature_spec.feature_dim,
        )
        if rknn_result.get("available"):
            rknn_output = np.asarray(rknn_result.pop("output"), dtype=np.float32)
            report["backends"]["rknn"] = {
                **rknn_result,
                "path": str(rknn_path),
                "input_shape": list(rknn_input.shape),
                "output_shape": list(rknn_output.shape),
                "error_vs_pytorch": _summary_error(torch_output[:rknn_frames], rknn_output),
            }
        else:
            report["backends"]["rknn"] = {
                **rknn_result,
                "path": str(rknn_path),
                "input_shape": list(rknn_input.shape),
                "note": "Run this script in the WSL RKNN environment or on a RKNN runtime machine for real RKNN numerical comparison.",
            }
    return report


def write_consistency_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path
