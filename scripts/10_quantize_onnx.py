from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.deploy.quantization import quantize_onnx_dynamic
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ONNX INT8 dynamic quantization.")
    parser.add_argument("--config", default="configs/quantize_onnx.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("quantize_onnx", cfg["logging"]["log_path"])
    output_path = quantize_onnx_dynamic(
        cfg["onnx"]["input_path"],
        cfg["onnx"]["output_path"],
        weight_type=str(cfg["quantization"]["weight_type"]),
        per_channel=bool(cfg["quantization"]["per_channel"]),
    )
    logger.info("Quantized ONNX model: %s", output_path)
    print(f"Quantized ONNX: {output_path}")


if __name__ == "__main__":
    main()
