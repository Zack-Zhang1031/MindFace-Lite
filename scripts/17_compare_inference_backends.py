from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.deploy.consistency import compare_backends_from_config, write_consistency_report
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch, ONNXRuntime, and optional RKNN outputs.")
    parser.add_argument("--config", default="configs/consistency_compare.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("consistency_compare", cfg["logging"]["log_path"])
    report = compare_backends_from_config(cfg)
    output_path = write_consistency_report(report, cfg["output"]["report_path"])
    logger.info("Consistency report: %s", output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()

