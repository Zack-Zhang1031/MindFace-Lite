from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.training.pruning import prune_and_finetune_from_config
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune a checkpoint and fine-tune it.")
    parser.add_argument("--config", default="configs/prune_finetune.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("prune_finetune", cfg["logging"]["log_path"])
    output_path = prune_and_finetune_from_config(cfg, logger)
    print(f"Pruned checkpoint: {output_path}")


if __name__ == "__main__":
    main()
