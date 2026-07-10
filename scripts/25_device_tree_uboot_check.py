from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def print_uboot_notes(dtb_name: str, overlay_name: str, boot_partition: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile/check an RK3588 Device Tree overlay and print U-Boot steps.")
    parser.add_argument("--config", default="configs/device_tree_uboot.yaml")
    parser.add_argument("--check-deps", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("device_tree_uboot_check", cfg["logging"]["log_path"])
    dtc_path = shutil.which("dtc")
    if args.check_deps:
        print(f"dtc: {dtc_path if dtc_path else 'not found'}")
        return

    dts_path = resolve_path(cfg["device_tree"]["dts_path"])
    dtbo_path = resolve_path(cfg["device_tree"]["dtbo_path"])
    if not dts_path.exists():
        raise FileNotFoundError(f"DTS file not found: {dts_path}")
    if dtc_path is None:
        print("dtc was not found in PATH. Install the Device Tree Compiler in the BSP/Linux environment.")
        print_uboot_notes(cfg["uboot"]["dtb_name"], cfg["uboot"]["overlay_name"], cfg["uboot"]["boot_partition"])
        return

    dtbo_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [dtc_path, "-@", "-I", "dts", "-O", "dtb", "-o", str(dtbo_path), str(dts_path)]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Compiled DTBO: %s", dtbo_path)
    print(f"DTBO: {dtbo_path}")
    print_uboot_notes(cfg["uboot"]["dtb_name"], cfg["uboot"]["overlay_name"], cfg["uboot"]["boot_partition"])


if __name__ == "__main__":
    main()
