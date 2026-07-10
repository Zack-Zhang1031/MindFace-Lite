from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mindface.utils.config import resolve_path


def rknn_environment_note() -> str:
    return (
        "RKNN-Toolkit2 is a Rockchip conversion SDK. It is normally installed in an Ubuntu x86_64 "
        "RKNN development environment, not in this Windows training environment. Official packages "
        "target Ubuntu 18.04/20.04/22.04/24.04 with matching Python versions. On the RK3588 board, "
        "use RKNN-Toolkit-Lite2 or RKNN Runtime for deployment."
    )


def check_rknn_available() -> tuple[bool, str]:
    ensure_onnx_mapping_compat()
    try:
        from rknn.api import RKNN  # noqa: F401
    except ImportError:
        system = platform.system()
        return (
            False,
            "rknn-toolkit2 is not installed in this Python environment. "
            f"Current platform: {system}. {rknn_environment_note()}",
        )
    return True, "rknn-toolkit2 is installed."


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _path_info(path: str | Path | None) -> dict:
    if path is None:
        return {"path": None, "exists": None, "size_bytes": None}
    resolved = resolve_path(path)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "size_bytes": resolved.stat().st_size if resolved.exists() and resolved.is_file() else None,
    }


def _dataset_info(path: str | Path | None) -> dict:
    info = _path_info(path)
    if not path or not info["exists"]:
        info["num_entries"] = None
        return info
    resolved = resolve_path(path)
    try:
        lines = [line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        lines = []
    info["num_entries"] = len(lines)
    info["first_entry"] = lines[0] if lines else None
    return info


def collect_rknn_environment() -> dict:
    ok, message = check_rknn_available()
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "onnx": _package_version("onnx"),
        "onnxruntime": _package_version("onnxruntime"),
        "rknn_toolkit2": _package_version("rknn-toolkit2"),
        "rknn_available": ok,
        "rknn_message": message,
    }


def ensure_onnx_mapping_compat() -> None:
    """Restore the old onnx.mapping namespace expected by RKNN-Toolkit2.

    ONNX 1.22 removed the public onnx.mapping module. RKNN-Toolkit2 2.3.2
    still calls onnx.mapping.TENSOR_TYPE_TO_NP_TYPE during load_onnx().
    """
    try:
        import onnx
    except ImportError:
        return
    if hasattr(onnx, "mapping"):
        return
    if not hasattr(onnx, "_mapping") or not hasattr(onnx._mapping, "TENSOR_TYPE_MAP"):
        return
    tensor_type_to_np_type = {
        tensor_type: dtype_info.np_dtype for tensor_type, dtype_info in onnx._mapping.TENSOR_TYPE_MAP.items()
    }
    np_type_to_tensor_type = {np.dtype(value): key for key, value in tensor_type_to_np_type.items()}
    onnx.mapping = SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE=tensor_type_to_np_type,
        NP_TYPE_TO_TENSOR_TYPE=np_type_to_tensor_type,
    )


def make_dry_run_report(
    onnx_path: str | Path,
    output_path: str | Path,
    target_platform: str,
    do_quantization: bool,
    dataset_path: str | Path | None,
    run_inference: bool,
) -> dict:
    onnx_path = resolve_path(onnx_path)
    output_path = resolve_path(output_path)
    dataset_resolved = resolve_path(dataset_path) if dataset_path else None
    ok, message = check_rknn_available()
    return {
        "dry_run": True,
        "environment": collect_rknn_environment(),
        "rknn_available": ok,
        "rknn_message": message,
        "onnx_model": _path_info(onnx_path),
        "rknn_output": _path_info(output_path),
        "target_platform": target_platform,
        "do_quantization": do_quantization,
        "quantization_dataset": _dataset_info(dataset_resolved),
        "run_inference": run_inference,
        "expected_steps": [
            "Install RKNN-Toolkit2 in Ubuntu x86_64 or Rockchip SDK Docker.",
            "Run rknn.config(target_platform='rk3588').",
            "Run rknn.load_onnx(model=...).",
            "Run rknn.build(do_quantization=...).",
            "Run rknn.export_rknn(...).",
            "Copy the .rknn model to the RK3588 board.",
            "Run board inference with RKNN-Toolkit-Lite2, RKNN Runtime, or the C++ runtime API.",
        ],
    }


def create_feature_calibration_dataset(
    output_dir: str | Path,
    num_samples: int,
    frames: int,
    feature_dim: int,
    seed: int,
) -> Path:
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    list_path = output_dir / "rknn_calibration_dataset.txt"
    rows = []
    for idx in range(num_samples):
        sample = rng.random((frames, feature_dim), dtype=np.float32)
        sample_path = output_dir / f"calib_{idx:04d}.npy"
        np.save(sample_path, sample)
        rows.append(str(sample_path))
    list_path.write_text("\n".join(rows), encoding="utf-8")
    return list_path


def convert_onnx_to_rknn(
    onnx_path: str | Path,
    output_path: str | Path,
    target_platform: str,
    do_quantization: bool,
    dataset_path: str | Path | None,
    verbose: bool,
    runtime_target: str | None,
    run_inference: bool,
    dummy_frames: int,
    feature_dim: int,
) -> dict:
    ensure_onnx_mapping_compat()
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is not installed. Install it in the RKNN development environment, then rerun this script."
        ) from exc

    onnx_path = resolve_path(onnx_path)
    output_path = resolve_path(output_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if do_quantization and dataset_path is None:
        raise ValueError("RKNN quantization requires a calibration dataset path.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=verbose)
    report: dict[str, object] = {
        "environment": collect_rknn_environment(),
        "onnx_model": _path_info(onnx_path),
        "rknn_output": _path_info(output_path),
        "target_platform": target_platform,
        "do_quantization": do_quantization,
        "quantization_dataset": _dataset_info(dataset_path),
        "input": {
            "name": "audio_features",
            "frames": int(dummy_frames),
            "feature_dim": int(feature_dim),
            "shape": [int(dummy_frames), int(feature_dim)],
        },
        "notes": [
            "This script performs conversion in the x86_64 RKNN development environment.",
            "Board-side inference still needs RKNN Runtime or RKNN-Toolkit-Lite2 on an RK3588 device.",
            "If INT8 quantization is enabled, replace dummy calibration data with representative audio features.",
        ],
    }
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"RKNN config failed ret={ret}")
        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["audio_features"],
            input_size_list=[[int(dummy_frames), int(feature_dim)]],
        )
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed ret={ret}")
        build_kwargs = {"do_quantization": bool(do_quantization)}
        if do_quantization:
            build_kwargs["dataset"] = str(resolve_path(dataset_path))
        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed ret={ret}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"RKNN export_rknn failed ret={ret}")
        report["exported"] = True
        report["rknn_output"] = _path_info(output_path)

        if run_inference:
            ret = rknn.init_runtime(target=runtime_target) if runtime_target else rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"RKNN init_runtime failed ret={ret}")
            dummy = np.random.default_rng(42).random((dummy_frames, feature_dim), dtype=np.float32)
            outputs = rknn.inference(inputs=[dummy])
            report["runtime_target"] = runtime_target or "local"
            report["inference_output_shape"] = list(np.asarray(outputs[0]).shape)
        report["next_steps"] = [
            "Copy the exported .rknn file to the RK3588 board.",
            "Install or link RKNN Runtime / RKNN-Toolkit-Lite2 on the board.",
            "Feed the same audio feature shape used during conversion.",
            "Record board latency, CPU usage, DDR pressure, heat, and output consistency.",
        ]
    finally:
        rknn.release()
    return report
