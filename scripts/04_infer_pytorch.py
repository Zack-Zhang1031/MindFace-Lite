from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.inference import predict_from_wav
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.csv_io import write_params_csv
from mindface.utils.logger import setup_logger
from mindface.visual.mouth_drawer import params_to_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyTorch mouth inference from WAV.")
    parser.add_argument("--config", default="configs/infer_pytorch.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("infer_pytorch", cfg["logging"]["log_path"])
    times, params = predict_from_wav(
        cfg["checkpoint"]["path"],
        cfg["audio"]["input_path"],
        fps=int(cfg["audio"]["fps"]),
        frame_ms=float(cfg["audio"]["frame_ms"]),
        smooth_alpha=float(cfg["inference"]["smooth_alpha"]),
        device=str(cfg["inference"]["device"]),
    )
    write_params_csv(cfg["csv"]["output_path"], times, params)
    params_to_video(
        params,
        cfg["video"]["output_path"],
        fps=int(cfg["audio"]["fps"]),
        width=int(cfg["video"]["width"]),
        height=int(cfg["video"]["height"]),
    )
    logger.info("Wrote PyTorch inference outputs.")
    print(f"CSV: {resolve_path(cfg['csv']['output_path'])}")
    print(f"Video: {resolve_path(cfg['video']['output_path'])}")


if __name__ == "__main__":
    main()
