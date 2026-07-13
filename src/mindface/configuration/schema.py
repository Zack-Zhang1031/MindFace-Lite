from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindface.utils.config import load_yaml, resolve_path


@dataclass(frozen=True, slots=True)
class ValidationReport:
    path: Path
    schema_name: str
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


_REQUIRED_BY_FILE: dict[str, tuple[str, ...]] = {
    "rule-demo.yaml": ("audio", "mouth", "video", "csv", "logging"),
    "better-visual-demo.yaml": ("audio", "mouth", "video", "csv", "logging"),
    "expressive-avatar-demo.yaml": (
        "asset",
        "audio",
        "mouth",
        "mouth_roi",
        "expressive",
        "viseme",
        "video",
        "preview",
        "csv",
        "logging",
    ),
    "synthetic-dataset.yaml": ("dataset", "audio", "features"),
    "prepare-grid.yaml": ("raw_grid_dir", "output_dir", "scan", "features", "labels", "split", "output"),
    "grid-video-landmarks.yaml": ("grid", "landmarks", "logging", "quality"),
    "prepare-grid-landmark.yaml": (
        "raw_grid_dir",
        "landmark_dir",
        "output_dir",
        "min_detection_rate",
        "features",
        "output",
    ),
    "infer-pytorch.yaml": ("audio", "checkpoint", "inference", "video", "csv", "logging"),
    "infer-onnx.yaml": ("audio", "onnx", "inference", "video", "csv", "logging"),
    "quantize-onnx.yaml": ("onnx", "quantization", "logging"),
    "quantize-grid-onnx.yaml": ("onnx", "quantization", "logging"),
    "prune-finetune.yaml": ("checkpoint", "dataset", "pruning", "train", "logging"),
    "benchmark.yaml": ("checkpoint", "onnx", "benchmark", "output", "logging"),
    "benchmark-quantized-onnx.yaml": ("onnx", "benchmark", "output", "logging"),
    "benchmark-grid-quantized-onnx.yaml": ("onnx", "benchmark", "output", "logging"),
    "benchmark-pruned.yaml": ("checkpoints", "benchmark", "output", "logging"),
    "backend-consistency.yaml": (
        "audio",
        "checkpoint",
        "pytorch",
        "onnx",
        "rknn",
        "compare",
        "output",
        "logging",
    ),
    "realtime-rule.yaml": ("audio", "realtime", "output", "logging"),
    "mic-stream.yaml": ("audio", "mouth", "output", "udp", "logging"),
    "tts-demo.yaml": ("text", "audio", "rule_config"),
    "real-tts.yaml": ("engine", "text", "audio", "pyttsx3", "edge_tts", "logging", "rule_config", "outputs"),
    "export-onnx.yaml": ("checkpoint", "onnx", "logging"),
    "export-grid-onnx.yaml": ("checkpoint", "onnx", "logging"),
    "rknn-deploy.yaml": ("model", "quantization", "input", "runtime", "output", "logging"),
    "device-tree-uboot.yaml": ("device_tree", "uboot", "logging"),
}

_TRAIN_REQUIRED = (
    "dataset.dir",
    "features.fps",
    "features.frame_ms",
    "model.type",
    "model.params",
    "train.epochs",
    "train.batch_size",
    "train.lr",
    "output.checkpoint_path",
    "output.log_path",
)


def _value_at(data: dict[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def _require(data: dict[str, Any], paths: tuple[str, ...], errors: list[str]) -> None:
    for dotted_path in paths:
        try:
            value = _value_at(data, dotted_path)
        except KeyError:
            errors.append(f"missing required field: {dotted_path}")
            continue
        if value is None or value == "":
            errors.append(f"field must not be empty: {dotted_path}")


def _positive(data: dict[str, Any], dotted_path: str, errors: list[str]) -> None:
    try:
        value = _value_at(data, dotted_path)
    except KeyError:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        errors.append(f"field must be a positive number: {dotted_path}")


def _validate_ratios(data: dict[str, Any], errors: list[str]) -> None:
    split = data.get("split")
    if isinstance(split, dict) and all(key in split for key in ("train_ratio", "val_ratio", "test_ratio")):
        ratios = [split["train_ratio"], split["val_ratio"], split["test_ratio"]]
        if not all(isinstance(value, (int, float)) and value >= 0 for value in ratios):
            errors.append("split ratios must be non-negative numbers")
        elif abs(sum(float(value) for value in ratios) - 1.0) > 1e-6:
            errors.append("split ratios must sum to 1.0")

    grid = data.get("grid")
    if isinstance(grid, dict) and "split_ratios" in grid:
        ratios = grid["split_ratios"]
        if not isinstance(ratios, list) or len(ratios) != 3:
            errors.append("grid.split_ratios must contain train, val, and test ratios")
        elif not all(isinstance(value, (int, float)) and value >= 0 for value in ratios):
            errors.append("grid split ratios must be non-negative numbers")
        elif abs(sum(float(value) for value in ratios) - 1.0) > 1e-6:
            errors.append("grid split ratios must sum to 1.0")


def _schema_for(path: Path, schema_name: str | None) -> str:
    if schema_name is not None:
        return schema_name
    parent = path.parent.name
    if parent in {"demos", "datasets", "training", "inference", "optimization", "benchmarks", "realtime", "deployment"}:
        return parent
    return "unknown"


def validate_config(path: str | Path, schema_name: str | None = None) -> ValidationReport:
    resolved = resolve_path(path)
    schema = _schema_for(resolved, schema_name)
    errors: list[str] = []
    try:
        data = load_yaml(resolved)
    except (FileNotFoundError, ValueError) as exc:
        return ValidationReport(resolved, schema, (str(exc),))

    if schema == "training":
        _require(data, _TRAIN_REQUIRED, errors)
        for field in ("features.fps", "features.frame_ms", "train.epochs", "train.batch_size", "train.lr"):
            _positive(data, field, errors)
        try:
            params = _value_at(data, "model.params")
            if not isinstance(params, dict):
                errors.append("field must be a mapping: model.params")
        except KeyError:
            pass
    else:
        required = _REQUIRED_BY_FILE.get(resolved.name)
        if required is None:
            errors.append(f"no schema registered for config file: {resolved.name}")
        else:
            _require(data, required, errors)

    for candidate in ("features.fps", "features.frame_ms", "features.feature_dim", "audio.fps", "audio.frame_ms"):
        _positive(data, candidate, errors)
    _validate_ratios(data, errors)
    return ValidationReport(resolved, schema, tuple(errors))


def validate_all_configs(root: str | Path = "configs") -> list[ValidationReport]:
    config_root = resolve_path(root)
    return [validate_config(path) for path in sorted(config_root.rglob("*.yaml"))]

