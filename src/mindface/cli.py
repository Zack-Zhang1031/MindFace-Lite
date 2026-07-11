from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mindface.utils.config import project_root


def _run_script(script_name: str, args: list[str]) -> int:
    command = [sys.executable, str(project_root() / "scripts" / script_name), *args]
    return subprocess.run(command, cwd=project_root(), check=False).returncode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mindface", description="MindFace-Lite unified CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Run project health check.")
    subparsers.add_parser("rule-demo", help="Run Stage 1 rule mouth demo.")
    subparsers.add_parser("better-visual", help="Run Stage 1.5 better visual renderer.")
    subparsers.add_parser("expressive-avatar", help="Run Stage 1.6 static avatar mouth-warp demo.")
    subparsers.add_parser("compare-backends", help="Compare PyTorch, ONNXRuntime, and optional RKNN outputs.")

    train = subparsers.add_parser("train", help="Train from a YAML config.")
    train.add_argument("--config", default="configs/train_mlp.yaml")

    export = subparsers.add_parser("export-onnx", help="Export checkpoint to ONNX.")
    export.add_argument("--config", default="configs/export_onnx.yaml")

    prepare = subparsers.add_parser("prepare-grid-landmark", help="Prepare GRID landmark supervised dataset.")
    prepare.add_argument("--config", default="configs/prepare_grid_landmark.yaml")
    prepare.add_argument("--max-samples", default=None)

    args = parser.parse_args(argv)
    if args.command == "prepare-grid-landmark":
        script_args = ["--config", args.config]
        if args.max_samples is not None:
            script_args.extend(["--max-samples", str(args.max_samples)])
        code = _run_script("16_prepare_grid_landmark_dataset.py", script_args)
    elif args.command == "train":
        code = _run_script("03_train_model.py", ["--config", args.config])
    elif args.command == "export-onnx":
        code = _run_script("05_export_onnx.py", ["--config", args.config])
    else:
        mapping = {
            "health": ("99_health_check.py", []),
            "rule-demo": ("01_rule_mouth_demo.py", ["--config", "configs/rule_demo.yaml"]),
            "better-visual": ("01_5_better_visual_demo.py", ["--config", "configs/better_visual_demo.yaml"]),
            "expressive-avatar": ("01_6_expressive_avatar_demo.py", ["--config", "configs/expressive_avatar_demo.yaml"]),
            "compare-backends": ("17_compare_inference_backends.py", ["--config", "configs/consistency_compare.yaml"]),
        }
        script, script_args = mapping[args.command]
        code = _run_script(script, script_args)
    raise SystemExit(code)
