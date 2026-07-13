from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mindface.utils.config import project_root, resolve_path


@dataclass(frozen=True, slots=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    outputs: tuple[str | Path, ...] = ()
    optional: bool = False


@dataclass(frozen=True, slots=True)
class PipelineResult:
    name: str
    status: str
    command: tuple[str, ...]
    return_code: int | None = None


def select_steps(
    steps: Sequence[PipelineStep],
    from_step: str | None = None,
    to_step: str | None = None,
) -> list[PipelineStep]:
    names = [step.name for step in steps]
    for requested in (from_step, to_step):
        if requested is not None and requested not in names:
            raise ValueError(f"Unknown pipeline step '{requested}'. Available: {', '.join(names)}")
    start = names.index(from_step) if from_step is not None else 0
    end = names.index(to_step) if to_step is not None else len(steps) - 1
    if start > end:
        raise ValueError(f"Pipeline from-step '{from_step}' comes after to-step '{to_step}'")
    return list(steps[start : end + 1])


def _default_runner(command: list[str]) -> int:
    env = os.environ.copy()
    source_path = str(project_root() / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_path, env.get("PYTHONPATH"))))
    return subprocess.run(command, cwd=project_root(), check=False, env=env).returncode


def _outputs_exist(step: PipelineStep) -> bool:
    return bool(step.outputs) and all(resolve_path(path).exists() for path in step.outputs)


def run_pipeline(
    steps: Sequence[PipelineStep],
    *,
    dry_run: bool = False,
    force: bool = False,
    runner: Callable[[list[str]], int] = _default_runner,
    python_executable: str = sys.executable,
) -> list[PipelineResult]:
    results: list[PipelineResult] = []
    for step in steps:
        command = [python_executable, *step.command]
        if dry_run:
            results.append(PipelineResult(step.name, "planned", tuple(command)))
            continue
        if not force and _outputs_exist(step):
            results.append(PipelineResult(step.name, "skipped", tuple(command)))
            continue
        return_code = runner(command)
        if return_code != 0 and not step.optional:
            raise RuntimeError(f"Pipeline step '{step.name}' failed with code {return_code}: {' '.join(command)}")
        status = "completed" if return_code == 0 else "optional-failed"
        results.append(PipelineResult(step.name, status, tuple(command), return_code))
    return results


def optional_dependency_steps() -> list[PipelineStep]:
    return [
        PipelineStep("check-landmarks", ("-m", "mindface", "data", "extract-landmarks", "--check-deps"), optional=True),
        PipelineStep("check-tts", ("-m", "mindface", "tts", "generate", "--check-deps"), optional=True),
        PipelineStep("check-microphone", ("-m", "mindface", "realtime", "microphone", "--check-deps"), optional=True),
        PipelineStep("check-rknn", ("-m", "mindface", "deploy", "rknn", "--check-deps"), optional=True),
        PipelineStep("check-device-tree", ("-m", "mindface", "deploy", "device-tree", "--check-deps"), optional=True),
    ]


def basic_pipeline_steps(
    *,
    skip_quantization: bool = False,
    include_grid_compression: bool = False,
    check_optional_deps: bool = False,
    check_optional_deps_only: bool = False,
) -> list[PipelineStep]:
    if check_optional_deps_only:
        return optional_dependency_steps()
    steps = [
        PipelineStep("generate-audio", ("-m", "mindface", "demo", "generate-audio"), ("outputs/audio/test_voice.wav",)),
        PipelineStep(
            "rule-demo",
            ("-m", "mindface", "demo", "rule", "--config", "configs/demos/rule-demo.yaml"),
            ("outputs/videos/rule_mouth_demo.mp4",),
        ),
        PipelineStep(
            "better-visual",
            ("-m", "mindface", "demo", "better-visual", "--config", "configs/demos/better-visual-demo.yaml"),
            ("outputs/videos/better_visual_mouth_demo.mp4",),
        ),
        PipelineStep(
            "expressive-avatar",
            ("-m", "mindface", "demo", "expressive-avatar", "--config", "configs/demos/expressive-avatar-demo.yaml"),
            ("outputs/videos/expressive_avatar_demo.mp4",),
        ),
        PipelineStep(
            "synthetic-data",
            ("-m", "mindface", "data", "synthetic", "--config", "configs/datasets/synthetic-dataset.yaml"),
            ("data/synthetic_mouth/manifest.csv",),
        ),
        PipelineStep(
            "train",
            ("-m", "mindface", "train", "--config", "configs/training/train-mlp.yaml"),
            ("outputs/checkpoints/mlp_mouth.pt",),
        ),
        PipelineStep(
            "infer-pytorch",
            ("-m", "mindface", "infer", "pytorch", "--config", "configs/inference/infer-pytorch.yaml"),
            ("outputs/videos/pytorch_mlp_demo.mp4",),
        ),
        PipelineStep(
            "export-onnx",
            ("-m", "mindface", "export", "onnx", "--config", "configs/deployment/export-onnx.yaml"),
            ("outputs/models/mlp_mouth.onnx",),
        ),
        PipelineStep(
            "infer-onnx",
            ("-m", "mindface", "infer", "onnx", "--config", "configs/inference/infer-onnx.yaml"),
            ("outputs/videos/onnx_mlp_demo.mp4",),
        ),
        PipelineStep(
            "compare-backends",
            ("-m", "mindface", "benchmark", "backends", "--config", "configs/benchmarks/backend-consistency.yaml"),
            ("outputs/reports/backend_consistency_report.json",),
        ),
        PipelineStep(
            "realtime-queue",
            ("-m", "mindface", "realtime", "queue", "--config", "configs/realtime/realtime-rule.yaml"),
            ("outputs/reports/realtime_rule_report.json",),
        ),
        PipelineStep(
            "benchmark",
            ("-m", "mindface", "benchmark", "runtime", "--config", "configs/benchmarks/benchmark.yaml"),
            ("outputs/reports/benchmark_report.json",),
        ),
    ]
    if not skip_quantization:
        steps.extend(
            [
                PipelineStep(
                    "quantize",
                    ("-m", "mindface", "optimize", "quantize", "--config", "configs/optimization/quantize-onnx.yaml"),
                    ("outputs/models/mlp_mouth.int8.dynamic.onnx",),
                ),
                PipelineStep(
                    "benchmark-quantized",
                    (
                        "-m",
                        "mindface",
                        "optimize",
                        "benchmark-quantized",
                        "--config",
                        "configs/benchmarks/benchmark-quantized-onnx.yaml",
                    ),
                    ("outputs/reports/quantized_onnx_benchmark.json",),
                ),
            ]
        )
    if include_grid_compression:
        steps.extend(
            [
                PipelineStep(
                    "export-grid-onnx",
                    ("-m", "mindface", "export", "onnx", "--config", "configs/deployment/export-grid-onnx.yaml"),
                    ("outputs/models/grid_mlp_mouth.onnx",),
                ),
                PipelineStep(
                    "quantize-grid",
                    ("-m", "mindface", "optimize", "quantize", "--config", "configs/optimization/quantize-grid-onnx.yaml"),
                    ("outputs/models/grid_mlp_mouth.int8.dynamic.onnx",),
                ),
                PipelineStep(
                    "benchmark-grid-quantized",
                    (
                        "-m",
                        "mindface",
                        "optimize",
                        "benchmark-quantized",
                        "--config",
                        "configs/benchmarks/benchmark-grid-quantized-onnx.yaml",
                    ),
                    ("outputs/reports/grid_quantized_onnx_benchmark.json",),
                ),
                PipelineStep(
                    "prune-grid",
                    ("-m", "mindface", "optimize", "prune", "--config", "configs/optimization/prune-finetune.yaml"),
                    ("outputs/checkpoints/grid_mlp_mouth.pruned.pt",),
                ),
                PipelineStep(
                    "benchmark-pruned",
                    ("-m", "mindface", "optimize", "benchmark-pruned", "--config", "configs/benchmarks/benchmark-pruned.yaml"),
                    ("outputs/reports/pruned_benchmark.json",),
                ),
            ]
        )
    if check_optional_deps:
        steps.extend(optional_dependency_steps())
    return steps
