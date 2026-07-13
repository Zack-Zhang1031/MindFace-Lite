from __future__ import annotations

import argparse

import numpy as np

from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.csv_io import write_params_csv
from mindface.utils.logger import setup_logger


def run_train(args: argparse.Namespace) -> int:
    from mindface.training.trainer import train_from_config

    cfg = load_yaml(args.config)
    logger = setup_logger("train_model", cfg["output"]["log_path"])
    checkpoint_path = train_from_config(cfg, logger)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {resolve_path(cfg['output']['log_path'])}")
    return 0


def run_infer_pytorch(args: argparse.Namespace) -> int:
    from mindface.inference import predict_from_wav
    from mindface.visual.mouth_drawer import params_to_video

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
    return 0


def run_infer_onnx(args: argparse.Namespace) -> int:
    from mindface.audio.features import extract_audio_features, read_wav_mono, smooth_params
    from mindface.deploy.onnx_tools import run_onnx_inference
    from mindface.visual.mouth_drawer import params_to_video

    cfg = load_yaml(args.config)
    logger = setup_logger("infer_onnx", cfg["logging"]["log_path"])
    sample_rate, waveform = read_wav_mono(cfg["audio"]["input_path"])
    fps = int(cfg["audio"]["fps"])
    features = extract_audio_features(
        waveform,
        sample_rate,
        fps=fps,
        frame_ms=float(cfg["audio"]["frame_ms"]),
    )
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
    return 0


def run_export_onnx(args: argparse.Namespace) -> int:
    from mindface.deploy.onnx_tools import export_checkpoint_to_onnx

    cfg = load_yaml(args.config)
    logger = setup_logger("export_onnx", cfg["logging"]["log_path"])
    output_path = export_checkpoint_to_onnx(
        cfg["checkpoint"]["path"],
        cfg["onnx"]["output_path"],
        opset=int(cfg["onnx"]["opset"]),
    )
    logger.info("Exported ONNX: %s", output_path)
    print(f"ONNX: {output_path}")
    return 0
