from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.audio.features import extract_audio_features, read_wav_mono, smooth_params
from mindface.deploy.onnx_tools import run_onnx_inference
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.csv_io import write_params_csv
from mindface.utils.logger import setup_logger
from mindface.visual.mouth_drawer import params_to_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNXRuntime mouth inference from WAV.")
    parser.add_argument("--config", default="configs/infer_onnx.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("infer_onnx", cfg["logging"]["log_path"])
    sample_rate, waveform = read_wav_mono(cfg["audio"]["input_path"])
    fps = int(cfg["audio"]["fps"])
    features = extract_audio_features(waveform, sample_rate, fps=fps, frame_ms=float(cfg["audio"]["frame_ms"]))
    params = run_onnx_inference(cfg["onnx"]["path"], features, model_type=str(cfg["onnx"]["model_type"]))
    params = smooth_params(params, alpha=float(cfg["inference"]["smooth_alpha"]))
    times = np.arange(len(params), dtype=np.float32) / float(fps)
    write_params_csv(cfg["csv"]["output_path"], times, params)
    params_to_video(
        params,
        cfg["video"]["output_path"],
        fps=fps,
        width=int(cfg["video"]["width"]),
        height=int(cfg["video"]["height"]),
    )
    logger.info("Wrote ONNX inference outputs.")
    print(f"CSV: {resolve_path(cfg['csv']['output_path'])}")
    print(f"Video: {resolve_path(cfg['video']['output_path'])}")


if __name__ == "__main__":
    main()
