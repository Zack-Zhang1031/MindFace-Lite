from __future__ import annotations

import argparse
from typing import Any

from mindface.commands import run_command


def _add_leaf(
    subparsers: Any,
    name: str,
    command_id: str,
    help_text: str,
    default_config: str | None = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=help_text)
    parser.set_defaults(command_id=command_id)
    if default_config is not None:
        parser.add_argument("--config", default=default_config)
    return parser


def _add_group(subparsers: Any, name: str, help_text: str) -> Any:
    parser = subparsers.add_parser(name, help=help_text)
    return parser.add_subparsers(dest=f"{name}_command", required=True)


def _add_visual_overrides(parser: argparse.ArgumentParser, debug_flag: str) -> None:
    parser.add_argument("--input-wav", default=None)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(debug_flag, action="store_true")


def _add_tts_options(parser: argparse.ArgumentParser, real_backend: bool) -> None:
    parser.add_argument("--text", default=None)
    if real_backend:
        parser.add_argument("--engine", choices=["pyttsx3", "edge_tts"], default=None)
        parser.add_argument("--check-deps", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mindface", description="MindFace-Lite unified CLI.")
    commands = parser.add_subparsers(dest="command", required=True)

    ui = commands.add_parser("ui", help="Open the interactive learning workbench.")
    ui.set_defaults(handler=_run_ui)

    health = _add_leaf(commands, "health", "health", "Run project health checks.")
    health.add_argument("--output", default="outputs/reports/health_check.json")
    health.add_argument("--json-only", action="store_true")

    config = _add_group(commands, "config", "List, inspect, and validate YAML configs.")
    config_list = config.add_parser("list", help="List project YAML configs.")
    config_list.set_defaults(handler=_run_config_command, config_action="list")
    config_show = config.add_parser("show", help="Print a resolved YAML config.")
    config_show.add_argument("path")
    config_show.set_defaults(handler=_run_config_command, config_action="show")
    config_validate = config.add_parser("validate", help="Validate one or all YAML configs.")
    config_validate.add_argument("path", nargs="?", default=None)
    config_validate.add_argument("--all", action="store_true", dest="validate_all")
    config_validate.set_defaults(handler=_run_config_command, config_action="validate")

    environment = _add_group(commands, "env", "Inspect, install, and repair project environments.")
    for name, command_id, full_help in (
        ("status", "env.status", "Show lightweight Windows and WSL environment status."),
        ("check", "env.check", "Run full Windows and WSL dependency checks."),
    ):
        environment_check = _add_leaf(environment, name, command_id, full_help)
        environment_check.add_argument("--distro", default="Ubuntu")
        environment_check.add_argument("--output", default="outputs/reports/environment-matrix.json")
    install_windows = _add_leaf(
        environment,
        "install-windows",
        "env.install-windows",
        "Create or repair the Windows Conda training environment.",
    )
    install_windows.add_argument("--source", choices=["official", "tsinghua", "aliyun"], default="official")
    install_windows.add_argument("--yes", action="store_true")
    install_windows.add_argument("--dry-run", action="store_true")
    install_windows.add_argument("--log-path", default="outputs/logs/environment-install-windows.log")
    install_wsl = _add_leaf(
        environment,
        "install-wsl",
        "env.install-wsl",
        "Create or repair the Ubuntu RKNN environment.",
    )
    install_wsl.add_argument("--distro", default="Ubuntu")
    install_wsl.add_argument("--source", choices=["official", "tsinghua", "aliyun"], default="official")
    install_wsl.add_argument("--yes", action="store_true")
    install_wsl.add_argument("--dry-run", action="store_true")
    install_wsl.add_argument("--log-path", default="outputs/logs/environment-install-wsl.log")

    pipeline = _add_group(commands, "pipeline", "Run staged pipelines.")
    basic = _add_leaf(pipeline, "basic", "pipeline.basic", "Run the basic validation pipeline.")
    basic.add_argument("--skip-quantization", action="store_true")
    basic.add_argument("--include-grid-compression", action="store_true")
    basic.add_argument("--check-optional-deps", action="store_true")
    basic.add_argument("--check-optional-deps-only", action="store_true")
    basic.add_argument("--dry-run", action="store_true")
    basic.add_argument("--force", action="store_true")
    basic.add_argument("--from-step", default=None)
    basic.add_argument("--to-step", default=None)
    _add_leaf(commands, "verify", "verify", "Verify generated basic outputs.")

    demos = _add_group(commands, "demo", "Generate audio and run visual demos.")
    generated_audio = _add_leaf(demos, "generate-audio", "demo.generate-audio", "Generate test WAV audio.")
    generated_audio.add_argument("--output", default="outputs/audio/test_voice.wav")
    generated_audio.add_argument("--sample-rate", type=int, default=16000)
    generated_audio.add_argument("--duration-sec", type=float, default=4.2)
    _add_leaf(
        demos,
        "rule",
        "demo.rule",
        "Run the Stage 1 rule demo.",
        "configs/demos/rule-demo.yaml",
    )
    better = _add_leaf(
        demos,
        "better-visual",
        "demo.better-visual",
        "Run the Stage 1.5 renderer.",
        "configs/demos/better-visual-demo.yaml",
    )
    _add_visual_overrides(better, "--debug-overlay")
    expressive = _add_leaf(
        demos,
        "expressive-avatar",
        "demo.expressive-avatar",
        "Run the Stage 1.6 expressive avatar.",
        "configs/demos/expressive-avatar-demo.yaml",
    )
    _add_visual_overrides(expressive, "--debug-roi")

    data = _add_group(commands, "data", "Generate and prepare datasets.")
    _add_leaf(
        data,
        "synthetic",
        "data.synthetic",
        "Generate the synthetic training dataset.",
        "configs/datasets/synthetic-dataset.yaml",
    )
    prepare_grid = _add_leaf(
        data,
        "prepare-grid",
        "data.prepare-grid",
        "Prepare GRID audio with rule targets.",
        "configs/datasets/prepare-grid.yaml",
    )
    prepare_grid.add_argument("--max-samples", type=int, default=None)
    prepare_grid.add_argument("--output-dir", default=None)
    landmarks = _add_leaf(
        data,
        "extract-landmarks",
        "data.extract-landmarks",
        "Extract GRID video mouth landmarks.",
        "configs/datasets/grid-video-landmarks.yaml",
    )
    landmarks.add_argument("--max-videos", type=int, default=None)
    landmarks.add_argument("--output-dir", default=None)
    landmarks.add_argument("--delegate", choices=["cpu", "gpu"], default=None)
    landmarks.add_argument("--check-deps", action="store_true")
    landmarks.add_argument("--quality-only", action="store_true")
    align = _add_leaf(
        data,
        "align-landmarks",
        "data.align-landmarks",
        "Align GRID audio features with landmark targets.",
        "configs/datasets/prepare-grid-landmark.yaml",
    )
    align.add_argument("--max-samples", type=int, default=None)

    _add_leaf(
        commands,
        "train",
        "train",
        "Train from a YAML config.",
        "configs/training/train-mlp.yaml",
    )

    infer = _add_group(commands, "infer", "Run model inference.")
    _add_leaf(
        infer,
        "pytorch",
        "infer.pytorch",
        "Run PyTorch inference.",
        "configs/inference/infer-pytorch.yaml",
    )
    _add_leaf(
        infer,
        "onnx",
        "infer.onnx",
        "Run ONNXRuntime inference.",
        "configs/inference/infer-onnx.yaml",
    )

    export = _add_group(commands, "export", "Export deployable models.")
    _add_leaf(
        export,
        "onnx",
        "export.onnx",
        "Export a checkpoint to ONNX.",
        "configs/deployment/export-onnx.yaml",
    )

    optimize = _add_group(commands, "optimize", "Quantize, prune, and compare optimized models.")
    _add_leaf(
        optimize,
        "quantize",
        "optimize.quantize",
        "Apply ONNX dynamic quantization.",
        "configs/optimization/quantize-onnx.yaml",
    )
    _add_leaf(
        optimize,
        "benchmark-quantized",
        "optimize.benchmark-quantized",
        "Compare FP32 and quantized ONNX.",
        "configs/benchmarks/benchmark-quantized-onnx.yaml",
    )
    _add_leaf(
        optimize,
        "prune",
        "optimize.prune",
        "Prune and fine-tune a checkpoint.",
        "configs/optimization/prune-finetune.yaml",
    )
    _add_leaf(
        optimize,
        "benchmark-pruned",
        "optimize.benchmark-pruned",
        "Compare original and pruned checkpoints.",
        "configs/benchmarks/benchmark-pruned.yaml",
    )

    benchmarks = _add_group(commands, "benchmark", "Run performance and consistency benchmarks.")
    _add_leaf(
        benchmarks,
        "runtime",
        "benchmark.runtime",
        "Benchmark PyTorch and ONNXRuntime.",
        "configs/benchmarks/benchmark.yaml",
    )
    _add_leaf(
        benchmarks,
        "backends",
        "benchmark.backends",
        "Compare PyTorch, ONNXRuntime, and optional RKNN.",
        "configs/benchmarks/backend-consistency.yaml",
    )

    realtime = _add_group(commands, "realtime", "Run realtime simulations and microphone input.")
    _add_leaf(
        realtime,
        "queue",
        "realtime.queue",
        "Run the queue-based realtime simulation.",
        "configs/realtime/realtime-rule.yaml",
    )
    microphone = _add_leaf(
        realtime,
        "microphone",
        "realtime.microphone",
        "Run microphone RMS mouth control.",
        "configs/realtime/mic-stream.yaml",
    )
    microphone.add_argument("--duration-sec", type=float, default=None)
    microphone.add_argument("--show", action="store_true")
    microphone.add_argument("--check-deps", action="store_true")
    microphone.add_argument("--list-devices", action="store_true")

    tts = _add_group(commands, "tts", "Generate TTS audio and mouth demos.")
    pseudo_generate = _add_leaf(
        tts,
        "pseudo-generate",
        "tts.pseudo-generate",
        "Generate deterministic TTS-like audio.",
        "configs/realtime/tts-demo.yaml",
    )
    _add_tts_options(pseudo_generate, real_backend=False)
    pseudo_demo = _add_leaf(
        tts,
        "pseudo-demo",
        "tts.pseudo-demo",
        "Drive a mouth demo with TTS-like audio.",
        "configs/realtime/tts-demo.yaml",
    )
    _add_tts_options(pseudo_demo, real_backend=False)
    real_generate = _add_leaf(
        tts,
        "generate",
        "tts.generate",
        "Generate audio with a real TTS backend.",
        "configs/realtime/real-tts.yaml",
    )
    _add_tts_options(real_generate, real_backend=True)
    real_demo = _add_leaf(
        tts,
        "demo",
        "tts.demo",
        "Drive a mouth demo with real TTS audio.",
        "configs/realtime/real-tts.yaml",
    )
    _add_tts_options(real_demo, real_backend=True)

    deploy = _add_group(commands, "deploy", "Run RKNN and embedded deployment tools.")
    rknn = _add_leaf(
        deploy,
        "rknn",
        "deploy.rknn",
        "Convert ONNX to RKNN and optionally infer.",
        "configs/deployment/rknn-deploy.yaml",
    )
    rknn.add_argument("--quantize", action="store_true")
    rknn.add_argument("--run-inference", action="store_true")
    rknn.add_argument("--check-deps", action="store_true")
    rknn.add_argument("--dry-run", action="store_true")
    rknn.add_argument("--report-path", default=None)
    device_tree = _add_leaf(
        deploy,
        "device-tree",
        "deploy.device-tree",
        "Check Device Tree and cross-compile tools.",
        "configs/deployment/device-tree-uboot.yaml",
    )
    device_tree.add_argument("--check-deps", action="store_true")

    cpp = _add_group(commands, "cpp", "Configure, build, test, and run C++ demos.")
    for name, command_id, help_text, presets in (
        ("configure", "cpp.configure", "Configure the CMake project.", ["windows-release", "linux-release", "arm64-release"]),
        ("build", "cpp.build", "Build the configured C++ project.", ["windows-release", "linux-release", "arm64-release"]),
        ("test", "cpp.test", "Run CTest for the C++ project.", ["windows-release", "linux-release"]),
    ):
        cpp_command = _add_leaf(cpp, name, command_id, help_text)
        cpp_command.add_argument(
            "--preset",
            choices=presets,
            default=None,
        )
    cpp_run = _add_leaf(cpp, "run", "cpp.run", "Run a built C++ demo executable.")
    cpp_run.add_argument("--preset", default=None)
    cpp_run.add_argument(
        "app",
        choices=["queue-demo", "udp-sender", "serial-sender", "onnxruntime-cpp-demo"],
    )
    cpp_run.add_argument("app_args", nargs=argparse.REMAINDER)

    project = _add_group(commands, "project", "Run project-level verification tools.")
    project_test = _add_leaf(project, "test", "project.test", "Run the Python test suite.")
    project_test.add_argument("--quick", action="store_true")
    _add_leaf(project, "compile", "project.compile", "Compile-check Python sources.")

    # Compatibility aliases for commands documented before the grouped CLI.
    _add_leaf(
        commands,
        "rule-demo",
        "demo.rule",
        "Compatibility alias for 'demo rule'.",
        "configs/demos/rule-demo.yaml",
    )
    old_better = _add_leaf(
        commands,
        "better-visual",
        "demo.better-visual",
        "Compatibility alias for 'demo better-visual'.",
        "configs/demos/better-visual-demo.yaml",
    )
    _add_visual_overrides(old_better, "--debug-overlay")
    old_expressive = _add_leaf(
        commands,
        "expressive-avatar",
        "demo.expressive-avatar",
        "Compatibility alias for 'demo expressive-avatar'.",
        "configs/demos/expressive-avatar-demo.yaml",
    )
    _add_visual_overrides(old_expressive, "--debug-roi")
    _add_leaf(
        commands,
        "compare-backends",
        "benchmark.backends",
        "Compatibility alias for 'benchmark backends'.",
        "configs/benchmarks/backend-consistency.yaml",
    )
    _add_leaf(
        commands,
        "export-onnx",
        "export.onnx",
        "Compatibility alias for 'export onnx'.",
        "configs/deployment/export-onnx.yaml",
    )
    old_align = _add_leaf(
        commands,
        "prepare-grid-landmark",
        "data.align-landmarks",
        "Compatibility alias for 'data align-landmarks'.",
        "configs/datasets/prepare-grid-landmark.yaml",
    )
    old_align.add_argument("--max-samples", type=int, default=None)
    return parser


def _run_config_command(args: argparse.Namespace) -> int:
    import yaml

    from mindface.configuration.schema import validate_all_configs, validate_config
    from mindface.utils.config import load_yaml, project_root, resolve_path

    if args.config_action == "list":
        for path in sorted(resolve_path("configs").rglob("*.yaml")):
            print(path.relative_to(project_root()).as_posix())
        return 0
    if args.config_action == "show":
        print(yaml.safe_dump(load_yaml(args.path), sort_keys=False, allow_unicode=True), end="")
        return 0
    if args.validate_all:
        reports = validate_all_configs()
    elif args.path is not None:
        reports = [validate_config(args.path)]
    else:
        raise ValueError("config validate requires PATH or --all")
    for report in reports:
        status = "VALID" if report.valid else "INVALID"
        print(f"[{status}] {report.path.relative_to(project_root()) if report.path.is_relative_to(project_root()) else report.path}")
        for error in report.errors:
            print(f"  - {error}")
    valid_count = sum(report.valid for report in reports)
    print(f"{valid_count} valid, {len(reports) - valid_count} invalid")
    return 0 if valid_count == len(reports) else 1


def _run_ui(args: argparse.Namespace) -> int:
    del args
    from mindface.terminal_ui import launch

    return launch(run)


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is not None:
        return int(handler(args))
    return run_command(args.command_id, args)


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))
