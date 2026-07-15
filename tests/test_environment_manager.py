from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mindface.environment.manager import (
    build_windows_install_plan,
    build_wsl_install_plan,
    execute_plan,
    pip_source_args,
)


def test_windows_plan_reuses_existing_environment_without_delete(tmp_path: Path) -> None:
    plan = build_windows_install_plan(
        root=tmp_path,
        conda_executable="conda",
        environment_exists=True,
        source="official",
    )
    rendered = "\n".join(" ".join(step.command) for step in plan.steps)

    assert "conda create" not in rendered
    assert "remove" not in rendered
    assert "requirements.txt" in rendered
    assert "requirements-optional.txt" in rendered
    assert "requirements-dev.txt" in rendered
    assert "pip check" in rendered


def test_windows_plan_creates_missing_python_310_environment(tmp_path: Path) -> None:
    plan = build_windows_install_plan(
        root=tmp_path,
        conda_executable="conda",
        environment_exists=False,
        source="official",
    )

    assert plan.steps[0].command == (
        "conda",
        "create",
        "-n",
        "mindface-lite",
        "python=3.10",
        "-y",
    )


def test_wsl_plan_keeps_rknn_environment_isolated(tmp_path: Path) -> None:
    plan = build_wsl_install_plan(root=tmp_path, distro="Ubuntu", source="official")
    rendered = "\n".join(" ".join(step.command) for step in plan.steps)

    assert "sudo apt install" in rendered
    assert "requirements-rknn.txt" in rendered
    assert "requirements-optional.txt" not in rendered
    assert "~/.venvs/mindface-rknn" in rendered
    assert "rm -rf" not in rendered


def test_pip_download_source_is_selected_per_install() -> None:
    assert pip_source_args("official") == ()
    assert pip_source_args("tsinghua") == (
        "--index-url",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
    )
    assert pip_source_args("aliyun") == (
        "--index-url",
        "https://mirrors.aliyun.com/pypi/simple",
    )


def test_execute_plan_stops_after_first_failure(tmp_path: Path) -> None:
    plan = build_windows_install_plan(
        root=tmp_path,
        conda_executable="conda",
        environment_exists=True,
        source="official",
    )
    calls: list[tuple[str, ...]] = []

    def fail_first(command: list[str], cwd: Path) -> int:
        del cwd
        calls.append(tuple(command))
        return 7

    result = execute_plan(plan, log_path=tmp_path / "install.log", runner=fail_first)

    assert result.success is False
    assert result.return_code == 7
    assert len(calls) == 1
    assert (tmp_path / "install.log").exists()


def test_environment_cli_bootstraps_without_site_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "mindface",
            "env",
            "install-windows",
            "--dry-run",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Environment installation plan: windows" in completed.stdout
