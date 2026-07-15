from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mindface.environment.manager import (
    EnvironmentPlan,
    build_windows_install_plan,
    build_wsl_install_plan,
    execute_plan,
    inspect_environment_matrix,
    list_wsl_distros,
    render_plan,
)

ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _print_environment_report(report: dict) -> None:
    summary = report["summary"]
    mode = "full" if report["full"] else "status"
    print(
        f"Environment {mode}: pass={summary.get('pass', 0)} "
        f"warn={summary.get('warn', 0)} fail={summary.get('fail', 0)}"
    )
    for check in report["checks"]:
        print(f"[{check['status'].upper():4}] {check['name']}: {check['message']}")


def _run_environment_report(args: argparse.Namespace, *, full: bool) -> int:
    report = inspect_environment_matrix(root=ROOT, distro=args.distro, full=full)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_environment_report(report)
    print(f"Report: {output_path}")
    return 0


def run_status(args: argparse.Namespace) -> int:
    return _run_environment_report(args, full=False)


def run_check(args: argparse.Namespace) -> int:
    return _run_environment_report(args, full=True)


def _confirm_install() -> bool:
    answer = input("\n确认执行以上全部步骤？现有环境不会被删除。[y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _execute_install(plan: EnvironmentPlan, args: argparse.Namespace) -> int:
    print(render_plan(plan))
    if args.dry_run:
        print("\nDry-run only; no installation command was executed.")
        return 0
    if not args.yes and not _confirm_install():
        print("Installation cancelled.")
        return 0
    result = execute_plan(plan, log_path=_resolve_path(args.log_path))
    if not result.success:
        print(
            f"Installation stopped at '{result.failed_step}' with code {result.return_code}. "
            f"Log: {result.log_path}"
        )
        return result.return_code or 1
    print(f"Environment installation completed. Log: {result.log_path}")
    return 0


def run_install_windows(args: argparse.Namespace) -> int:
    plan = build_windows_install_plan(root=ROOT, source=args.source)
    return _execute_install(plan, args)


def run_install_wsl(args: argparse.Namespace) -> int:
    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    if wsl is None:
        raise RuntimeError("wsl.exe is not in PATH. Enable WSL before installing the RKNN environment.")
    distros = list_wsl_distros(wsl)
    if args.distro not in distros:
        raise RuntimeError(f"WSL distro '{args.distro}' was not found. Available: {', '.join(distros) or 'none'}")
    plan = build_wsl_install_plan(
        root=ROOT,
        distro=args.distro,
        source=args.source,
        wsl_executable=wsl,
    )
    return _execute_install(plan, args)
