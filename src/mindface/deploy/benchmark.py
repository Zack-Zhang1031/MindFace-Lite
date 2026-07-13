from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from mindface.artifacts.model_bundle import load_model_bundle
from mindface.backends import OnnxPredictor, PyTorchPredictor
from mindface.utils.config import resolve_path


def latency_summary(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        raise ValueError("values_ms must not be empty")
    values = sorted(values_ms)
    mean_ms = float(np.mean(values))
    return {
        "mean_ms": mean_ms,
        "median_ms": float(values[len(values) // 2]),
        "p95_ms": float(values[min(len(values) - 1, int(len(values) * 0.95))]),
        "fps": float(1000.0 / max(mean_ms, 1e-9)),
    }


def _measure(predictor: Any, features: np.ndarray, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        predictor.predict(features)
    values: list[float] = []
    for _ in range(repeat):
        started = perf_counter()
        predictor.predict(features)
        values.append((perf_counter() - started) * 1000.0)
    return latency_summary(values)


def benchmark_from_config(cfg: dict[str, Any], logger: Any) -> Path:
    bench_cfg = cfg["benchmark"]
    bundle = load_model_bundle(cfg["checkpoint"]["path"])
    feature_dim = bundle.feature_spec.feature_dim
    configured_dim = int(bench_cfg["feature_dim"])
    if configured_dim != feature_dim:
        raise ValueError(f"Benchmark feature_dim={configured_dim} does not match checkpoint feature_dim={feature_dim}")
    features = np.random.default_rng(int(bench_cfg["seed"])).random(
        (int(bench_cfg["frames"]), feature_dim),
        dtype=np.float32,
    )
    warmup = int(bench_cfg["warmup"])
    repeat = int(bench_cfg["repeat"])
    pytorch = PyTorchPredictor(bundle, device="cpu")
    report: dict[str, Any] = {"feature_spec": bundle.feature_spec.to_dict(), "pytorch": _measure(pytorch, features, warmup, repeat)}
    try:
        onnx = OnnxPredictor(cfg["onnx"]["path"], bundle.model_type, feature_dim)
        report["onnxruntime"] = _measure(onnx, features, warmup, repeat)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        report["onnxruntime_error"] = str(exc)

    report_path = resolve_path(cfg["output"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Benchmark report: %s", report_path)
    return report_path

