from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.deploy.onnx_tools import run_onnx_inference
from mindface.inference import load_torch_model, predict_from_features
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def _latency_summary(values_ms: list[float]) -> dict[str, float]:
    values = sorted(values_ms)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(values[len(values) // 2]),
        "p95_ms": float(values[min(len(values) - 1, int(len(values) * 0.95))]),
        "fps": float(1000.0 / max(np.mean(values), 1e-9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNXRuntime inference.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("benchmark", cfg["logging"]["log_path"])
    bench_cfg = cfg["benchmark"]
    rng = np.random.default_rng(int(bench_cfg["seed"]))
    features = rng.random((int(bench_cfg["frames"]), int(bench_cfg["feature_dim"])), dtype=np.float32)
    warmup = int(bench_cfg["warmup"])
    repeat = int(bench_cfg["repeat"])

    model, checkpoint = load_torch_model(cfg["checkpoint"]["path"], device="cpu")
    for _ in range(warmup):
        predict_from_features(model, checkpoint["model_type"], features, device="cpu")
    torch_ms = []
    for _ in range(repeat):
        start = perf_counter()
        predict_from_features(model, checkpoint["model_type"], features, device="cpu")
        torch_ms.append((perf_counter() - start) * 1000.0)

    report = {"pytorch": _latency_summary(torch_ms)}
    try:
        for _ in range(warmup):
            run_onnx_inference(cfg["onnx"]["path"], features, model_type=str(cfg["onnx"]["model_type"]))
        onnx_ms = []
        for _ in range(repeat):
            start = perf_counter()
            run_onnx_inference(cfg["onnx"]["path"], features, model_type=str(cfg["onnx"]["model_type"]))
            onnx_ms.append((perf_counter() - start) * 1000.0)
        report["onnxruntime"] = _latency_summary(onnx_ms)
    except Exception as exc:
        report["onnxruntime_error"] = str(exc)

    report_path = resolve_path(cfg["output"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Benchmark report: %s", report_path)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
