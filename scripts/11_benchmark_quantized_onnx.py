from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.deploy.quantization import benchmark_onnx_pair, write_json_report
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FP32 ONNX and INT8 dynamic ONNX.")
    parser.add_argument("--config", default="configs/benchmark_quantized_onnx.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("benchmark_quantized_onnx", cfg["logging"]["log_path"])
    report = benchmark_onnx_pair(
        cfg["onnx"]["fp32_path"],
        cfg["onnx"]["int8_path"],
        model_type=str(cfg["onnx"]["model_type"]),
        frames=int(cfg["benchmark"]["frames"]),
        feature_dim=int(cfg["benchmark"]["feature_dim"]),
        warmup=int(cfg["benchmark"]["warmup"]),
        repeat=int(cfg["benchmark"]["repeat"]),
        seed=int(cfg["benchmark"]["seed"]),
    )
    report_path = write_json_report(cfg["output"]["report_path"], report)
    logger.info("Quantized ONNX benchmark report: %s", report_path)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
