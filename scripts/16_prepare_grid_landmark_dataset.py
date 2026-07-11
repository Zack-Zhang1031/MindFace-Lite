from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.data.grid_supervised import prepare_grid_landmark_dataset
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GRID audio features with video-landmark mouth labels.")
    parser.add_argument("--config", default="configs/prepare_grid_landmark.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("prepare_grid_landmark", cfg["output"]["log_path"])
    manifest_path = prepare_grid_landmark_dataset(cfg, max_samples_override=args.max_samples)
    logger.info("Prepared GRID landmark supervised dataset: %s", manifest_path)
    print(f"Manifest: {manifest_path}")
    print("Train with: python scripts/03_train_model.py --config configs/train_grid_landmark_mlp.yaml")


if __name__ == "__main__":
    main()

