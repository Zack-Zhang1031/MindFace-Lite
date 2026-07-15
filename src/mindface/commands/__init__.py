from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable


_COMMAND_HANDLERS: dict[str, str] = {
    "health": "mindface.commands.project:run_health",
    "env.status": "mindface.commands.environment:run_status",
    "env.check": "mindface.commands.environment:run_check",
    "env.install-windows": "mindface.commands.environment:run_install_windows",
    "env.install-wsl": "mindface.commands.environment:run_install_wsl",
    "verify": "mindface.commands.project:run_verify",
    "pipeline.basic": "mindface.commands.project:run_basic_pipeline",
    "project.test": "mindface.commands.project:run_tests",
    "project.compile": "mindface.commands.project:run_compileall",
    "demo.generate-audio": "mindface.commands.demo:run_generate_audio",
    "demo.rule": "mindface.commands.demo:run_rule_demo",
    "demo.better-visual": "mindface.commands.demo:run_better_visual_demo",
    "demo.expressive-avatar": "mindface.commands.demo:run_expressive_avatar_demo",
    "data.synthetic": "mindface.commands.data:run_synthetic",
    "data.prepare-grid": "mindface.commands.data:run_prepare_grid",
    "data.extract-landmarks": "mindface.commands.data:run_extract_landmarks",
    "data.align-landmarks": "mindface.commands.data:run_align_landmarks",
    "train": "mindface.commands.model:run_train",
    "infer.pytorch": "mindface.commands.model:run_infer_pytorch",
    "infer.onnx": "mindface.commands.model:run_infer_onnx",
    "export.onnx": "mindface.commands.model:run_export_onnx",
    "optimize.quantize": "mindface.commands.optimization:run_quantize",
    "optimize.benchmark-quantized": "mindface.commands.optimization:run_benchmark_quantized",
    "optimize.prune": "mindface.commands.optimization:run_prune",
    "optimize.benchmark-pruned": "mindface.commands.optimization:run_benchmark_pruned",
    "benchmark.runtime": "mindface.commands.optimization:run_benchmark_runtime",
    "benchmark.backends": "mindface.commands.optimization:run_benchmark_backends",
    "realtime.queue": "mindface.commands.realtime:run_queue",
    "realtime.microphone": "mindface.commands.realtime:run_microphone",
    "tts.pseudo-generate": "mindface.commands.realtime:run_pseudo_tts_generate",
    "tts.pseudo-demo": "mindface.commands.realtime:run_pseudo_tts_demo",
    "tts.generate": "mindface.commands.realtime:run_real_tts_generate",
    "tts.demo": "mindface.commands.realtime:run_real_tts_demo",
    "deploy.rknn": "mindface.commands.deployment:run_rknn",
    "deploy.device-tree": "mindface.commands.deployment:run_device_tree",
    "cpp.configure": "mindface.commands.deployment:run_cpp_configure",
    "cpp.build": "mindface.commands.deployment:run_cpp_build",
    "cpp.test": "mindface.commands.deployment:run_cpp_test",
    "cpp.run": "mindface.commands.deployment:run_cpp_app",
}


def command_ids() -> frozenset[str]:
    return frozenset(_COMMAND_HANDLERS)


def run_command(command_id: str, args: argparse.Namespace) -> int:
    try:
        target = _COMMAND_HANDLERS[command_id]
    except KeyError as exc:
        raise ValueError(f"Unknown internal command: {command_id}") from exc
    module_name, function_name = target.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    handler: Callable[[argparse.Namespace], int | None] = getattr(module, function_name)
    return int(handler(args) or 0)
