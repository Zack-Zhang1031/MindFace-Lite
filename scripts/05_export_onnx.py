from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.deploy.onnx_tools import export_checkpoint_to_onnx
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    parser.add_argument("--config", default="configs/export_onnx.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("export_onnx", cfg["logging"]["log_path"])
    output_path = export_checkpoint_to_onnx(
        cfg["checkpoint"]["path"],
        cfg["onnx"]["output_path"],
        opset=int(cfg["onnx"]["opset"]),
    )
    logger.info("Exported ONNX: %s", output_path)
    print(f"ONNX: {output_path}")


if __name__ == "__main__":
    main()
