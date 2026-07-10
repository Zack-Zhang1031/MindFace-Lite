from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.training.trainer import train_from_config
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MindFace-Lite mouth model.")
    parser.add_argument("--config", default="configs/train_mlp.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("train_model", cfg["output"]["log_path"])
    checkpoint_path = train_from_config(cfg, logger)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {resolve_path(cfg['output']['log_path'])}")


if __name__ == "__main__":
    main()
