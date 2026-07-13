from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from mindface.utils.config import load_yaml, project_root, resolve_path
from mindface.utils.logger import setup_logger


def run_rknn(args: argparse.Namespace) -> int:
    from mindface.deploy.rknn_pipeline import run_rknn_pipeline
    from mindface.deploy.rknn_tools import check_rknn_available

    if args.check_deps:
        print(check_rknn_available()[1])
        return 0
    report, report_path = run_rknn_pipeline(
        load_yaml(args.config),
        quantize=args.quantize,
        run_inference=args.run_inference,
        dry_run=args.dry_run,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report_path is not None:
        print(f"Report: {report_path}")
    return 0


def _print_uboot_notes(dtb_name: str, overlay_name: str, boot_partition: str) -> None:
    print("\nU-Boot overlay apply example:")
    print(f"fatload mmc {boot_partition} ${{fdt_addr_r}} {dtb_name}")
    print(f"fatload mmc {boot_partition} ${{fdtoverlay_addr_r}} {overlay_name}")
    print("fdt addr ${fdt_addr_r}")
    print("fdt apply ${fdtoverlay_addr_r}")
    print("booti ${kernel_addr_r} - ${fdt_addr_r}")
    print("\nLinux-side UART checks:")
    print("dmesg | grep -i tty")
    print("ls -l /dev/ttyS* /dev/ttyFIQ*")
    print("stty -F /dev/ttyS2 115200")


def run_device_tree(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    logger = setup_logger("device_tree_uboot_check", cfg["logging"]["log_path"])
    dtc_path = shutil.which("dtc")
    if args.check_deps:
        print(f"dtc: {dtc_path if dtc_path else 'not found'}")
        return 0
    dts_path = resolve_path(cfg["device_tree"]["dts_path"])
    dtbo_path = resolve_path(cfg["device_tree"]["dtbo_path"])
    if not dts_path.exists():
        raise FileNotFoundError(f"DTS file not found: {dts_path}")
    if dtc_path is None:
        print("dtc was not found in PATH. Install it in the BSP/Linux environment.")
        _print_uboot_notes(
            cfg["uboot"]["dtb_name"],
            cfg["uboot"]["overlay_name"],
            cfg["uboot"]["boot_partition"],
        )
        return 0
    dtbo_path.parent.mkdir(parents=True, exist_ok=True)
    command = [dtc_path, "-@", "-I", "dts", "-O", "dtb", "-o", str(dtbo_path), str(dts_path)]
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)
    print(f"DTBO: {dtbo_path}")
    _print_uboot_notes(
        cfg["uboot"]["dtb_name"],
        cfg["uboot"]["overlay_name"],
        cfg["uboot"]["boot_partition"],
    )
    return 0


def _default_preset() -> str:
    return "windows-release" if os.name == "nt" else "linux-release"


def _preset(args: argparse.Namespace) -> str:
    return args.preset or _default_preset()


def _run_tool(command: list[str], *, cwd: Path) -> int:
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError(f"Required tool is not in PATH: {command[0]}")
    return subprocess.run([executable, *command[1:]], cwd=cwd, check=False).returncode


def run_cpp_configure(args: argparse.Namespace) -> int:
    return _run_tool(["cmake", "--preset", _preset(args)], cwd=project_root() / "cpp")


def run_cpp_build(args: argparse.Namespace) -> int:
    return _run_tool(["cmake", "--build", "--preset", _preset(args)], cwd=project_root() / "cpp")


def run_cpp_test(args: argparse.Namespace) -> int:
    return _run_tool(["ctest", "--preset", _preset(args)], cwd=project_root() / "cpp")


def _cpp_executable(app: str, preset: str) -> Path:
    app_name = app.replace("-", "_")
    build_dir = {
        "windows-release": project_root() / "build" / "cpp-windows" / "Release",
        "linux-release": project_root() / "build" / "cpp-linux",
        "arm64-release": project_root() / "build" / "cpp-arm64",
    }.get(preset)
    if build_dir is None:
        raise ValueError(f"Cannot infer executable directory for custom preset: {preset}")
    suffix = ".exe" if preset == "windows-release" else ""
    return build_dir / f"{app_name}{suffix}"


def run_cpp_app(args: argparse.Namespace) -> int:
    preset = _preset(args)
    executable = _cpp_executable(args.app, preset)
    if not executable.exists():
        raise FileNotFoundError(f"C++ executable not found: {executable}. Run 'mindface cpp build' first.")
    return subprocess.run([str(executable), *args.app_args], cwd=project_root(), check=False).returncode
