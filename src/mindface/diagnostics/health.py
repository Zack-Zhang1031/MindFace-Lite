from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mindface.utils.config import project_root, resolve_path


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() and path.is_file() else None


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def check_python() -> CheckResult:
    return CheckResult(
        name="python",
        status="pass",
        message=f"Python {platform.python_version()}",
        details={
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
        },
    )


def check_git_workspace() -> CheckResult:
    root = project_root()
    if (root / ".git").exists():
        return CheckResult("git", "pass", "Git repository detected.", {"root": str(root)})
    return CheckResult(
        "git",
        "warn",
        "No .git directory detected. Initialize git before treating this as a portfolio repository.",
        {"root": str(root)},
    )


def check_imports() -> list[CheckResult]:
    packages = [
        ("numpy", "numpy", True),
        ("cv2", "opencv-python", True),
        ("yaml", "PyYAML", True),
        ("torch", "torch", True),
        ("onnx", "onnx", True),
        ("onnxruntime", "onnxruntime", True),
        ("mediapipe", "mediapipe", False),
        ("pyttsx3", "pyttsx3", False),
        ("edge_tts", "edge-tts", False),
        ("sounddevice", "sounddevice", False),
        ("rknn", "rknn-toolkit2", False),
    ]
    results: list[CheckResult] = []
    for module_name, package_name, required in packages:
        available = _module_available(module_name)
        version = _package_version(package_name)
        if available:
            results.append(
                CheckResult(
                    name=f"import:{module_name}",
                    status="pass",
                    message=f"{module_name} available" + (f" ({version})" if version else ""),
                    details={"package": package_name, "version": version},
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"import:{module_name}",
                    status="fail" if required else "warn",
                    message=f"{module_name} is not installed.",
                    details={"package": package_name, "required_for_basic_pipeline": required},
                )
            )
    return results


def _major_version(version: str | None) -> int | None:
    if not version:
        return None
    head = version.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def check_dependency_policy() -> list[CheckResult]:
    results: list[CheckResult] = []
    numpy_version = _package_version("numpy")
    opencv_version = _package_version("opencv-python") or _package_version("opencv-contrib-python")

    numpy_major = _major_version(numpy_version)
    if numpy_major is not None and numpy_major >= 2:
        results.append(
            CheckResult(
                "dependency_policy:numpy",
                "warn",
                "Current NumPy is 2.x. MindFace-Lite training/RKNN compatibility docs recommend numpy<2.0.",
                {"version": numpy_version},
            )
        )
    else:
        results.append(
            CheckResult(
                "dependency_policy:numpy",
                "pass",
                "NumPy version is compatible with the project dependency policy.",
                {"version": numpy_version},
            )
        )

    opencv_major = _major_version(opencv_version)
    if opencv_major is not None and opencv_major >= 5:
        results.append(
            CheckResult(
                "dependency_policy:opencv",
                "warn",
                "Current OpenCV is 5.x. Project docs pin OpenCV to 4.11.0.86 to avoid NumPy/RKNN conflicts.",
                {"version": opencv_version},
            )
        )
    else:
        results.append(
            CheckResult(
                "dependency_policy:opencv",
                "pass",
                "OpenCV version is compatible with the project dependency policy.",
                {"version": opencv_version},
            )
        )
    return results


def check_torch_cuda() -> CheckResult:
    if not _module_available("torch"):
        return CheckResult("torch_cuda", "fail", "torch is not installed.")
    import torch

    cuda_available = bool(torch.cuda.is_available())
    details: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if cuda_available:
        details["device_count"] = torch.cuda.device_count()
        details["device_name_0"] = torch.cuda.get_device_name(0)
        return CheckResult("torch_cuda", "pass", f"CUDA ready: {details['device_name_0']}", details)
    return CheckResult(
        "torch_cuda",
        "warn",
        "CUDA is not available in this environment. Training will run on CPU unless config requests cuda and fails fast.",
        details,
    )


def check_project_paths() -> list[CheckResult]:
    paths = [
        ("requirements", "requirements.txt", True),
        ("requirements_optional", "requirements-optional.txt", False),
        ("requirements_rknn", "requirements-rknn.txt", False),
        ("rule_config", "configs/rule_demo.yaml", True),
        ("expressive_avatar_config", "configs/expressive_avatar_demo.yaml", True),
        ("expressive_avatar_asset", "assets/avatar/stage1_6_static_face.png", True),
        ("train_config", "configs/train_mlp.yaml", True),
        ("grid_train_config", "configs/train_grid_mlp.yaml", False),
        ("rknn_config", "configs/rknn_deploy.yaml", False),
        ("test_audio", "outputs/audio/test_voice.wav", False),
        ("rule_video", "outputs/videos/rule_mouth_demo.mp4", False),
        ("better_visual_video", "outputs/videos/better_visual_mouth_demo.mp4", False),
        ("expressive_avatar_video", "outputs/videos/expressive_avatar_demo.mp4", False),
        ("mlp_checkpoint", "outputs/checkpoints/mlp_mouth.pt", False),
        ("mlp_onnx", "outputs/models/mlp_mouth.onnx", False),
        ("mlp_rknn", "outputs/models/mlp_mouth.rk3588.rknn", False),
    ]
    results: list[CheckResult] = []
    for name, rel_path, required in paths:
        path = resolve_path(rel_path)
        exists = path.exists()
        results.append(
            CheckResult(
                name=f"path:{name}",
                status="pass" if exists else ("fail" if required else "warn"),
                message=f"{rel_path} {'exists' if exists else 'is missing'}",
                details={"path": str(path), "size_bytes": _file_size(path)},
            )
        )
    return results


def check_grid_data() -> list[CheckResult]:
    results: list[CheckResult] = []
    manifests = [
        ("grid_audio_manifest", "data/processed/grid_mouth/manifest.csv"),
        ("grid_video_landmark_manifest", "data/processed/grid_video_landmarks/manifest.csv"),
    ]
    for name, rel_path in manifests:
        path = resolve_path(rel_path)
        rows = _count_csv_rows(path)
        if rows is None:
            results.append(
                CheckResult(
                    name=f"data:{name}",
                    status="warn",
                    message=f"{rel_path} is missing.",
                    details={"path": str(path)},
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"data:{name}",
                    status="pass",
                    message=f"{rel_path} has {rows} rows.",
                    details={"path": str(path), "rows": rows},
                )
            )
    return results


def check_external_tools() -> list[CheckResult]:
    tools = [
        ("git", True),
        ("cmake", False),
        ("dtc", False),
        ("aarch64-linux-gnu-gcc", False),
        ("aarch64-linux-gnu-g++", False),
    ]
    results: list[CheckResult] = []
    for tool, required in tools:
        path = shutil.which(tool)
        results.append(
            CheckResult(
                name=f"tool:{tool}",
                status="pass" if path else ("fail" if required else "warn"),
                message=f"{tool}: {path if path else 'not found'}",
                details={"path": path},
            )
        )
    return results


def run_health_checks() -> dict[str, Any]:
    checks: list[CheckResult] = [check_python(), check_git_workspace()]
    checks.extend(check_imports())
    checks.extend(check_dependency_policy())
    checks.append(check_torch_cuda())
    checks.extend(check_project_paths())
    checks.extend(check_grid_data())
    checks.extend(check_external_tools())

    counts = {"pass": 0, "warn": 0, "fail": 0, "info": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "project_root": str(project_root()),
        "summary": counts,
        "checks": [asdict(check) for check in checks],
    }
