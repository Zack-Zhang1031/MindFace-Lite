from __future__ import annotations

import argparse
import compileall
import json
import subprocess
import sys

from mindface.utils.config import project_root, resolve_path


REQUIRED_BASIC_OUTPUTS = (
    "outputs/audio/test_voice.wav",
    "outputs/videos/rule_mouth_demo.mp4",
    "outputs/videos/better_visual_mouth_demo.mp4",
    "outputs/videos/expressive_avatar_demo.mp4",
    "outputs/videos/expressive_avatar_preview.png",
    "outputs/logs/rule_demo.csv",
    "outputs/logs/better_visual_rule_demo.csv",
    "outputs/logs/expressive_avatar_demo.csv",
    "data/synthetic_mouth/manifest.csv",
    "outputs/checkpoints/mlp_mouth.pt",
    "outputs/videos/pytorch_mlp_demo.mp4",
    "outputs/models/mlp_mouth.onnx",
    "outputs/videos/onnx_mlp_demo.mp4",
    "outputs/reports/backend_consistency_report.json",
    "outputs/reports/benchmark_report.json",
)


def _print_health_report(report: dict) -> None:
    summary = report["summary"]
    print(
        "MindFace-Lite health check: "
        f"pass={summary.get('pass', 0)} warn={summary.get('warn', 0)} fail={summary.get('fail', 0)}"
    )
    for check in report["checks"]:
        print(f"[{str(check['status']).upper():4}] {check['name']}: {check['message']}")


def run_health(args: argparse.Namespace) -> int:
    from mindface.diagnostics.health import run_health_checks

    report = run_health_checks()
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.json_only:
        _print_health_report(report)
        print(f"Report: {output_path}")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    del args
    missing = [path for path in REQUIRED_BASIC_OUTPUTS if not resolve_path(path).exists()]
    if missing:
        print("Missing outputs:")
        for path in missing:
            print(f"  - {path}")
        return 1
    print("All basic outputs exist.")
    return 0


def run_basic_pipeline(args: argparse.Namespace) -> int:
    from mindface.pipelines.basic import basic_pipeline_steps, run_pipeline, select_steps

    steps = basic_pipeline_steps(
        skip_quantization=args.skip_quantization,
        include_grid_compression=args.include_grid_compression,
        check_optional_deps=args.check_optional_deps,
        check_optional_deps_only=args.check_optional_deps_only,
    )
    selected = select_steps(steps, from_step=args.from_step, to_step=args.to_step)
    results = run_pipeline(selected, dry_run=args.dry_run, force=args.force)
    for result in results:
        print(f"[{result.status.upper()}] {result.name}: {' '.join(result.command)}")
    print("MindFace-Lite pipeline completed.")
    return 0


def run_tests(args: argparse.Namespace) -> int:
    command = [sys.executable, "-m", "pytest"]
    if args.quick:
        command.extend(["-q", "--ignore=tests/test_extracted_workflows.py"])
    return subprocess.run(command, cwd=project_root(), check=False).returncode


def run_compileall(args: argparse.Namespace) -> int:
    del args
    ok = compileall.compile_dir(project_root() / "src", quiet=1)
    ok = compileall.compile_dir(project_root() / "scripts", quiet=1) and ok
    print("Python compileall passed." if ok else "Python compileall failed.")
    return 0 if ok else 1
