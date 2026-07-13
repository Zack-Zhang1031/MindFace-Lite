from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mindface.artifacts.model_bundle import load_model_bundle
from mindface.audio.features import extract_audio_features, read_wav_mono, smooth_params
from mindface.models.factory import build_model, is_sequence_model
from mindface.utils.config import resolve_path


def load_torch_model(checkpoint_path: str | Path, device: str = "cpu"):
    bundle = load_model_bundle(checkpoint_path, map_location=device)
    model = build_model(bundle.model_type, bundle.model_config)
    model.load_state_dict(bundle.model_state)
    model.to(device)
    model.eval()
    return model, bundle.to_checkpoint()


def predict_from_features(model, model_type: str, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    with torch.no_grad():
        if is_sequence_model(model_type):
            x = torch.from_numpy(features[None, :, :]).to(device).float()
            y = model(x)[0]
        else:
            x = torch.from_numpy(features).to(device).float()
            y = model(x)
    return y.detach().cpu().numpy().astype(np.float32)


def predict_from_wav(
    checkpoint_path: str | Path,
    audio_path: str | Path,
    fps: int | None = None,
    frame_ms: float | None = None,
    smooth_alpha: float = 0.65,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model, checkpoint = load_torch_model(checkpoint_path, device=device)
    sample_rate, waveform = read_wav_mono(audio_path)
    feature_config = checkpoint["feature_spec"]
    resolved_fps = int(feature_config["fps"] if fps is None else fps)
    resolved_frame_ms = float(feature_config["frame_ms"] if frame_ms is None else frame_ms)
    features = extract_audio_features(
        waveform,
        sample_rate,
        fps=resolved_fps,
        frame_ms=resolved_frame_ms,
        feature_dim=int(feature_config["feature_dim"]),
    )
    params = predict_from_features(model, checkpoint["model_type"], features, device=device)
    params = smooth_params(params, smooth_alpha)
    times = np.arange(len(params), dtype=np.float32) / float(resolved_fps)
    return times, params
