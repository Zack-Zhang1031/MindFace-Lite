from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


WINDOWS_ENV_NAME = "mindface-lite"
WINDOWS_PYTHON_VERSION = "3.10"
WSL_VENV = "~/.venvs/mindface-rknn"
DOWNLOAD_SOURCES = ("official", "tsinghua", "aliyun")

_PIP_INDEXES = {
    "official": None,
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
}


@dataclass(frozen=True, slots=True)
class CommandStep:
    name: str
    description: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentPlan:
    target: str
    root: Path
    steps: tuple[CommandStep, ...]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    success: bool
    return_code: int
    completed_steps: int
    failed_step: str | None
    log_path: Path


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]


Runner = Callable[[list[str], Path], int]


def pip_source_args(source: str) -> tuple[str, ...]:
    try:
        index = _PIP_INDEXES[source]
    except KeyError as exc:
        raise ValueError(f"Unknown download source '{source}'. Choose from: {', '.join(DOWNLOAD_SOURCES)}") from exc
    return () if index is None else ("--index-url", index)


def _conda_env_names(conda_executable: str) -> set[str]:
    completed = subprocess.run(
        [conda_executable, "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return set()
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return set()
    return {Path(path).name for path in payload.get("envs", [])}


def conda_environment_exists(conda_executable: str, environment_name: str = WINDOWS_ENV_NAME) -> bool:
    return environment_name in _conda_env_names(conda_executable)


def _conda_run(conda_executable: str, *args: str) -> tuple[str, ...]:
    return (conda_executable, "run", "-n", WINDOWS_ENV_NAME, *args)


def build_windows_install_plan(
    *,
    root: Path,
    source: str,
    conda_executable: str | None = None,
    environment_exists: bool | None = None,
) -> EnvironmentPlan:
    conda = conda_executable or shutil.which("conda")
    if conda is None:
        raise RuntimeError("Conda is not in PATH. Install Miniconda/Anaconda or open an Anaconda PowerShell prompt.")
    exists = conda_environment_exists(conda) if environment_exists is None else environment_exists
    source_args = pip_source_args(source)
    root = root.resolve()
    steps: list[CommandStep] = []
    if not exists:
        steps.append(
            CommandStep(
                "create-environment",
                f"Create Conda environment {WINDOWS_ENV_NAME} with Python {WINDOWS_PYTHON_VERSION}.",
                (conda, "create", "-n", WINDOWS_ENV_NAME, f"python={WINDOWS_PYTHON_VERSION}", "-y"),
            )
        )
    steps.extend(
        [
            CommandStep(
                "upgrade-installer",
                "Upgrade pip and wheel in the Windows training environment.",
                _conda_run(conda, "python", "-m", "pip", "install", "--upgrade", *source_args, "pip", "wheel"),
            ),
            CommandStep(
                "install-core",
                "Install training, PyTorch, ONNX, and runtime dependencies.",
                _conda_run(
                    conda,
                    "python",
                    "-m",
                    "pip",
                    "install",
                    *source_args,
                    "-r",
                    str(root / "requirements.txt"),
                ),
            ),
            CommandStep(
                "install-optional",
                "Install MediaPipe, TTS, microphone, and audio dependencies.",
                _conda_run(
                    conda,
                    "python",
                    "-m",
                    "pip",
                    "install",
                    *source_args,
                    "-r",
                    str(root / "requirements-optional.txt"),
                ),
            ),
            CommandStep(
                "install-development",
                "Install pytest and development dependencies.",
                _conda_run(
                    conda,
                    "python",
                    "-m",
                    "pip",
                    "install",
                    *source_args,
                    "-r",
                    str(root / "requirements-dev.txt"),
                ),
            ),
            CommandStep(
                "install-project",
                "Install MindFace-Lite in editable mode.",
                _conda_run(conda, "python", "-m", "pip", "install", *source_args, "-e", str(root)),
            ),
            CommandStep(
                "pip-check",
                "Check the Windows Python dependency graph.",
                _conda_run(conda, "python", "-m", "pip", "check"),
            ),
            CommandStep(
                "health-check",
                "Run the MindFace-Lite health check in the Windows environment.",
                _conda_run(conda, "python", "-m", "mindface", "health"),
            ),
        ]
    )
    return EnvironmentPlan("windows", root, tuple(steps))


def _windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if drive:
        tail = resolved.as_posix().split(":", maxsplit=1)[1].lstrip("/")
        return f"/mnt/{drive}/{tail}"
    return resolved.as_posix()


def _wsl_command(distro: str, script: str, wsl_executable: str = "wsl.exe") -> tuple[str, ...]:
    return (wsl_executable, "-d", distro, "--", "bash", "-lc", script)


def build_wsl_install_plan(
    *,
    root: Path,
    distro: str,
    source: str,
    wsl_executable: str = "wsl.exe",
) -> EnvironmentPlan:
    source_args = " ".join(shlex.quote(arg) for arg in pip_source_args(source))
    source_prefix = f"{source_args} " if source_args else ""
    wsl_root = shlex.quote(_windows_path_to_wsl(root))
    steps = (
        CommandStep(
            "install-system-tools",
            "Install WSL build, Device Tree, Python venv, and ARM64 cross-compile tools.",
            _wsl_command(
                distro,
                "sudo apt update && sudo apt install -y git cmake make ninja-build pkg-config "
                "python3-venv device-tree-compiler gcc-aarch64-linux-gnu g++-aarch64-linux-gnu",
                wsl_executable,
            ),
        ),
        CommandStep(
            "create-rknn-venv",
            "Create the RKNN venv only when it does not already exist.",
            _wsl_command(
                distro,
                f"test -x {WSL_VENV}/bin/python || python3 -m venv {WSL_VENV}",
                wsl_executable,
            ),
        ),
        CommandStep(
            "upgrade-rknn-installer",
            "Upgrade pip and wheel while preserving the pinned RKNN setuptools version.",
            _wsl_command(
                distro,
                f"{WSL_VENV}/bin/python -m pip install --upgrade {source_prefix}pip wheel",
                wsl_executable,
            ),
        ),
        CommandStep(
            "install-rknn-requirements",
            "Install only requirements-rknn.txt in the isolated RKNN environment.",
            _wsl_command(
                distro,
                f"{WSL_VENV}/bin/python -m pip install {source_prefix}-r {wsl_root}/requirements-rknn.txt",
                wsl_executable,
            ),
        ),
        CommandStep(
            "install-rknn-project",
            "Install MindFace-Lite in editable mode inside WSL.",
            _wsl_command(
                distro,
                f"{WSL_VENV}/bin/python -m pip install {source_prefix}-e {wsl_root}",
                wsl_executable,
            ),
        ),
        CommandStep(
            "check-rknn-dependencies",
            "Check the RKNN Python dependency graph.",
            _wsl_command(distro, f"{WSL_VENV}/bin/python -m pip check", wsl_executable),
        ),
        CommandStep(
            "check-rknn-toolkit",
            "Verify RKNN-Toolkit2 can be imported.",
            _wsl_command(
                distro,
                f"cd {wsl_root} && {WSL_VENV}/bin/python -m mindface deploy rknn --check-deps",
                wsl_executable,
            ),
        ),
    )
    return EnvironmentPlan("wsl", root.resolve(), steps)


def render_plan(plan: EnvironmentPlan) -> str:
    lines = [f"Environment installation plan: {plan.target}"]
    for index, step in enumerate(plan.steps, start=1):
        lines.append(f"{index}. {step.description}")
        lines.append(f"   {subprocess.list2cmdline(list(step.command))}")
    return "\n".join(lines)


def _default_runner(command: list[str], cwd: Path) -> int:
    return subprocess.run(command, cwd=cwd, check=False).returncode


def execute_plan(
    plan: EnvironmentPlan,
    *,
    log_path: Path,
    runner: Runner = _default_runner,
) -> ExecutionResult:
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed_steps = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] target={plan.target}\n")
        for index, step in enumerate(plan.steps, start=1):
            command_text = subprocess.list2cmdline(list(step.command))
            message = f"[{index}/{len(plan.steps)}] {step.description}"
            print(f"\n{message}\n{command_text}")
            log.write(f"{message}\ncommand={command_text}\n")
            log.flush()
            return_code = runner(list(step.command), plan.root)
            log.write(f"return_code={return_code}\n")
            log.flush()
            if return_code != 0:
                return ExecutionResult(False, return_code, completed_steps, step.name, log_path)
            completed_steps += 1
    return ExecutionResult(True, 0, completed_steps, None, log_path)


def _decode_windows_output(value: bytes) -> str:
    if b"\x00" in value:
        return value.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    return value.decode(errors="replace")


def list_wsl_distros(wsl_executable: str = "wsl.exe") -> list[str]:
    completed = subprocess.run([wsl_executable, "--list", "--quiet"], capture_output=True, check=False)
    if completed.returncode != 0:
        return []
    return [line.strip() for line in _decode_windows_output(completed.stdout).splitlines() if line.strip()]


def _capture(command: list[str], root: Path) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
    stdout = _decode_windows_output(completed.stdout).strip()
    stderr = _decode_windows_output(completed.stderr).strip()
    parts = (stdout, stderr) if completed.returncode != 0 else (stdout or stderr,)
    output = "\n".join(part for part in parts if part)
    return completed.returncode, output


def _check(name: str, ok: bool, message: str, **details: Any) -> EnvironmentCheck:
    return EnvironmentCheck(name, "pass" if ok else "warn", message, details)


def _python_probe_command(python_prefix: tuple[str, ...]) -> list[str]:
    script = """import importlib.metadata as metadata
import json
import sys

names = ['numpy', 'opencv-python', 'torch', 'onnx', 'onnxruntime', 'mediapipe', 'rknn-toolkit2']
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
cuda = False
gpu = None
try:
    import torch
    cuda = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda else None
except Exception:
    pass
print(json.dumps({'python': sys.version.split()[0], 'versions': versions, 'cuda': cuda, 'gpu': gpu}))
"""
    return [*python_prefix, "-c", script]


def _safe_probe(command: list[str], root: Path) -> tuple[bool, dict[str, Any] | str]:
    code, output = _capture(command, root)
    if code != 0:
        return False, output
    try:
        return True, json.loads(output.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False, output


def _running_in_windows_environment() -> bool:
    return os.environ.get("CONDA_DEFAULT_ENV") == WINDOWS_ENV_NAME or Path(sys.prefix).name == WINDOWS_ENV_NAME


def inspect_environment_matrix(
    *,
    root: Path,
    distro: str = "Ubuntu",
    full: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    checks: list[EnvironmentCheck] = [
        EnvironmentCheck("host", "pass", f"Host: {platform.platform()}", {"platform": platform.platform()}),
    ]
    conda = shutil.which("conda")
    checks.append(_check("windows:conda", conda is not None, f"Conda: {conda or 'not found'}", path=conda))
    windows_exists = bool(conda and conda_environment_exists(conda))
    checks.append(
        _check(
            "windows:environment",
            windows_exists,
            f"Conda environment '{WINDOWS_ENV_NAME}': {'present' if windows_exists else 'missing'}",
            environment=WINDOWS_ENV_NAME,
        )
    )
    if full and conda and windows_exists:
        python_prefix = (
            (sys.executable,)
            if _running_in_windows_environment()
            else (conda, "run", "-n", WINDOWS_ENV_NAME, "python")
        )
        ok, payload = _safe_probe(
            _python_probe_command(python_prefix),
            root,
        )
        checks.append(
            _check(
                "windows:python-packages",
                ok,
                "Windows Python package probe passed." if ok else "Windows Python package probe failed.",
                result=payload,
            )
        )
        pip_prefix = (
            [sys.executable]
            if _running_in_windows_environment()
            else [conda, "run", "-n", WINDOWS_ENV_NAME, "python"]
        )
        pip_code, pip_output = _capture(
            [*pip_prefix, "-m", "pip", "check"],
            root,
        )
        checks.append(_check("windows:pip-check", pip_code == 0, pip_output or "No broken requirements found.", return_code=pip_code))

    wsl = shutil.which("wsl.exe") or shutil.which("wsl")
    checks.append(_check("wsl:command", wsl is not None, f"WSL command: {wsl or 'not found'}", path=wsl))
    distros = list_wsl_distros(wsl) if wsl else []
    distro_exists = distro in distros
    checks.append(_check("wsl:distro", distro_exists, f"WSL distro '{distro}': {'present' if distro_exists else 'missing'}", distros=distros))
    venv_exists = False
    if wsl and distro_exists:
        venv_code, venv_output = _capture(
            list(_wsl_command(distro, f"test -x {WSL_VENV}/bin/python && {WSL_VENV}/bin/python --version", wsl)),
            root,
        )
        venv_exists = venv_code == 0
        checks.append(_check("wsl:rknn-venv", venv_exists, venv_output or f"{WSL_VENV} is missing.", path=WSL_VENV))
    if full and wsl and distro_exists and venv_exists:
        rknn_code, rknn_output = _capture(
            list(_wsl_command(distro, f"{WSL_VENV}/bin/python -c \"from rknn.api import RKNN; print('RKNN import OK')\"", wsl)),
            root,
        )
        checks.append(_check("wsl:rknn-toolkit", rknn_code == 0, rknn_output or "RKNN import failed.", return_code=rknn_code))
        pip_code, pip_output = _capture(
            list(_wsl_command(distro, f"{WSL_VENV}/bin/python -m pip check", wsl)),
            root,
        )
        checks.append(_check("wsl:pip-check", pip_code == 0, pip_output or "No broken requirements found.", return_code=pip_code))
        for tool in ("cmake", "ninja", "dtc", "aarch64-linux-gnu-gcc", "aarch64-linux-gnu-g++"):
            code, output = _capture(list(_wsl_command(distro, f"command -v {shlex.quote(tool)}", wsl)), root)
            checks.append(_check(f"wsl:tool:{tool}", code == 0, f"{tool}: {output or 'not found'}", path=output or None))

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "full": full,
        "summary": counts,
        "checks": [asdict(check) for check in checks],
    }
