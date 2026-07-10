from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.training.trainer import train_from_config
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    cfg = load_yaml("configs/train_mlp.yaml")
    logger = setup_logger("train_mlp", cfg["output"]["log_path"])
    checkpoint_path = train_from_config(cfg, logger)
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
