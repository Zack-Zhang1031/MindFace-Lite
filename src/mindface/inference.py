from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mindface.audio.features import extract_audio_features, read_wav_mono, smooth_params
from mindface.models.factory import build_model, is_sequence_model
from mindface.utils.config import resolve_path


def load_torch_model(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint["model_type"]
    model = build_model(model_type, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


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
    fps: int = 25,
    frame_ms: float = 40.0,
    smooth_alpha: float = 0.65,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model, checkpoint = load_torch_model(checkpoint_path, device=device)
    sample_rate, waveform = read_wav_mono(audio_path)
    features = extract_audio_features(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
    params = predict_from_features(model, checkpoint["model_type"], features, device=device)
    params = smooth_params(params, smooth_alpha)
    times = np.arange(len(params), dtype=np.float32) / float(fps)
    return times, params
