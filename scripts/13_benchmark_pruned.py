from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.training.pruning import benchmark_checkpoints
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark original and pruned PyTorch checkpoints.")
    parser.add_argument("--config", default="configs/benchmark_pruned.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("benchmark_pruned", cfg["logging"]["log_path"])
    report = benchmark_checkpoints(
        cfg["checkpoints"],
        frames=int(cfg["benchmark"]["frames"]),
        feature_dim=int(cfg["benchmark"]["feature_dim"]),
        warmup=int(cfg["benchmark"]["warmup"]),
        repeat=int(cfg["benchmark"]["repeat"]),
        seed=int(cfg["benchmark"]["seed"]),
    )
    output_path = resolve_path(cfg["output"]["report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Pruned benchmark report: %s", output_path)
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
