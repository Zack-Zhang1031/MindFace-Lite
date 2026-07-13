from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


LEGACY_CONFIG_PATHS: dict[str, str] = {
    "configs/benchmark.yaml": "configs/benchmarks/benchmark.yaml",
    "configs/benchmark_grid_quantized_onnx.yaml": "configs/benchmarks/benchmark-grid-quantized-onnx.yaml",
    "configs/benchmark_pruned.yaml": "configs/benchmarks/benchmark-pruned.yaml",
    "configs/benchmark_quantized_onnx.yaml": "configs/benchmarks/benchmark-quantized-onnx.yaml",
    "configs/better_visual_demo.yaml": "configs/demos/better-visual-demo.yaml",
    "configs/consistency_compare.yaml": "configs/benchmarks/backend-consistency.yaml",
    "configs/device_tree_uboot.yaml": "configs/deployment/device-tree-uboot.yaml",
    "configs/export_grid_onnx.yaml": "configs/deployment/export-grid-onnx.yaml",
    "configs/export_onnx.yaml": "configs/deployment/export-onnx.yaml",
    "configs/expressive_avatar_demo.yaml": "configs/demos/expressive-avatar-demo.yaml",
    "configs/grid_video_landmarks.yaml": "configs/datasets/grid-video-landmarks.yaml",
    "configs/infer_onnx.yaml": "configs/inference/infer-onnx.yaml",
    "configs/infer_pytorch.yaml": "configs/inference/infer-pytorch.yaml",
    "configs/mic_stream.yaml": "configs/realtime/mic-stream.yaml",
    "configs/prepare_grid.yaml": "configs/datasets/prepare-grid.yaml",
    "configs/prepare_grid_landmark.yaml": "configs/datasets/prepare-grid-landmark.yaml",
    "configs/prune_finetune.yaml": "configs/optimization/prune-finetune.yaml",
    "configs/quantize_grid_onnx.yaml": "configs/optimization/quantize-grid-onnx.yaml",
    "configs/quantize_onnx.yaml": "configs/optimization/quantize-onnx.yaml",
    "configs/real_tts.yaml": "configs/realtime/real-tts.yaml",
    "configs/realtime_rule.yaml": "configs/realtime/realtime-rule.yaml",
    "configs/rknn_deploy.yaml": "configs/deployment/rknn-deploy.yaml",
    "configs/rule_demo.yaml": "configs/demos/rule-demo.yaml",
    "configs/synthetic_dataset.yaml": "configs/datasets/synthetic-dataset.yaml",
    "configs/train_grid_debug_mlp.yaml": "configs/training/train-grid-debug-mlp.yaml",
    "configs/train_grid_landmark_mlp.yaml": "configs/training/train-grid-landmark-mlp.yaml",
    "configs/train_grid_mlp.yaml": "configs/training/train-grid-mlp.yaml",
    "configs/train_lstm.yaml": "configs/training/train-lstm.yaml",
    "configs/train_mlp.yaml": "configs/training/train-mlp.yaml",
    "configs/train_tcn.yaml": "configs/training/train-tcn.yaml",
    "configs/train_transformer.yaml": "configs/training/train-transformer.yaml",
    "configs/tts_demo.yaml": "configs/realtime/tts-demo.yaml",
}


def project_root() -> Path:
    """Return the repository root for scripts launched from any directory."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    legacy_key = path.as_posix()
    if legacy_key in LEGACY_CONFIG_PATHS:
        path = Path(LEGACY_CONFIG_PATHS[legacy_key])
    return project_root() / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def ensure_parent(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
