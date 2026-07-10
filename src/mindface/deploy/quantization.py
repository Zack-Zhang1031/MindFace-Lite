from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mindface.deploy.onnx_tools import run_onnx_inference
from mindface.utils.config import resolve_path


def quantize_onnx_dynamic(
    input_onnx_path: str | Path,
    output_onnx_path: str | Path,
    weight_type: str = "qint8",
    per_channel: bool = False,
) -> Path:
    """Apply ONNXRuntime dynamic weight quantization to an ONNX model."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError("onnxruntime quantization is unavailable. Install onnxruntime first.") from exc

    input_onnx_path = resolve_path(input_onnx_path)
    output_onnx_path = resolve_path(output_onnx_path)
    if not input_onnx_path.exists():
        raise FileNotFoundError(f"Input ONNX file not found: {input_onnx_path}")
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    quant_type = QuantType.QInt8 if weight_type.lower() == "qint8" else QuantType.QUInt8
    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(output_onnx_path),
        weight_type=quant_type,
        per_channel=per_channel,
    )
    return output_onnx_path


def file_size_report(paths: dict[str, str | Path]) -> dict[str, float]:
    report: dict[str, float] = {}
    for name, path in paths.items():
        resolved = resolve_path(path)
        report[name] = float(resolved.stat().st_size) if resolved.exists() else 0.0
    return report


def _latency_summary(values_ms: list[float]) -> dict[str, float]:
    values = sorted(values_ms)
    mean = float(np.mean(values)) if values else 0.0
    return {
        "mean_ms": mean,
        "median_ms": float(values[len(values) // 2]) if values else 0.0,
        "p95_ms": float(values[min(len(values) - 1, int(len(values) * 0.95))]) if values else 0.0,
        "fps": float(1000.0 / max(mean, 1e-9)),
    }


def benchmark_onnx_pair(
    fp32_onnx_path: str | Path,
    int8_onnx_path: str | Path,
    model_type: str,
    frames: int,
    feature_dim: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    features = rng.random((frames, feature_dim), dtype=np.float32)

    fp32_output = run_onnx_inference(fp32_onnx_path, features, model_type=model_type)
    int8_output = run_onnx_inference(int8_onnx_path, features, model_type=model_type)
    diff = np.abs(fp32_output - int8_output)

    timings = {}
    for name, path in {"fp32": fp32_onnx_path, "int8_dynamic": int8_onnx_path}.items():
        for _ in range(warmup):
            run_onnx_inference(path, features, model_type=model_type)
        values = []
        for _ in range(repeat):
            start = perf_counter()
            run_onnx_inference(path, features, model_type=model_type)
            values.append((perf_counter() - start) * 1000.0)
        timings[name] = _latency_summary(values)

    sizes = file_size_report({"fp32_bytes": fp32_onnx_path, "int8_dynamic_bytes": int8_onnx_path})
    return {
        "model_type": model_type,
        "input_shape": [frames, feature_dim],
        "sizes": sizes,
        "size_ratio_int8_over_fp32": sizes["int8_dynamic_bytes"] / max(sizes["fp32_bytes"], 1.0),
        "output_diff": {
            "mae": float(np.mean(diff)),
            "max_abs": float(np.max(diff)),
        },
        "latency": timings,
    }


def write_json_report(path: str | Path, report: dict) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
