from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from mindface.utils.config import resolve_path


def read_wav_mono(audio_path: str | Path) -> tuple[int, np.ndarray]:
    """Read a 16-bit PCM WAV file and return mono float32 audio in [-1, 1]."""
    audio_path = resolve_path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with wave.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample_width={sample_width}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def write_wav_mono(audio_path: str | Path, sample_rate: int, waveform: np.ndarray) -> None:
    """Write mono float audio in [-1, 1] to a 16-bit PCM WAV file."""
    audio_path = resolve_path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
    waveform = np.clip(waveform, -1.0, 1.0)
    audio_i16 = (waveform * 32767.0).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_i16.tobytes())


def frame_signal(waveform: np.ndarray, sample_rate: int, fps: int, frame_ms: float) -> list[np.ndarray]:
    hop = max(1, int(sample_rate / fps))
    size = max(1, int(sample_rate * frame_ms / 1000.0))
    total = max(1, int(np.ceil(len(waveform) / hop)))
    frames: list[np.ndarray] = []
    for idx in range(total):
        start = idx * hop
        frame = waveform[start : start + size]
        if len(frame) < size:
            frame = np.pad(frame, (0, size - len(frame)))
        frames.append(frame.astype(np.float32))
    return frames


def compute_rms(waveform: np.ndarray, sample_rate: int, fps: int = 25, frame_ms: float = 40.0) -> np.ndarray:
    """Compute RMS energy for each video frame.

    RMS formula for one frame x[0..N-1]:
    sqrt((x_0^2 + x_1^2 + ... + x_(N-1)^2) / N)
    """
    values = []
    for frame in frame_signal(waveform, sample_rate, fps, frame_ms):
        values.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)) + 1e-12)))
    return np.asarray(values, dtype=np.float32)


def rms_to_mouth_open(
    rms: np.ndarray,
    noise_floor: float = 0.01,
    scale: float | None = None,
    gamma: float = 0.65,
    smoothing: float = 0.35,
) -> np.ndarray:
    """Map RMS values to mouth_open in [0, 1] with normalization and smoothing."""
    rms = np.asarray(rms, dtype=np.float32)
    if scale is None:
        scale = float(np.percentile(rms, 95)) if rms.size else 1.0
    scale = max(scale - noise_floor, 1e-6)
    values = np.clip((rms - noise_floor) / scale, 0.0, 1.0)
    values = np.power(values, gamma)
    if smoothing > 0:
        out = np.zeros_like(values)
        prev = 0.0
        for idx, value in enumerate(values):
            prev = smoothing * prev + (1.0 - smoothing) * float(value)
            out[idx] = prev
        values = out
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def extract_audio_features(
    waveform: np.ndarray,
    sample_rate: int,
    fps: int = 25,
    frame_ms: float = 40.0,
    feature_dim: int = 70,
) -> np.ndarray:
    """Extract dependency-light frame features for training and inference.

    The default 70-D feature vector is intentionally simple: time-domain stats,
    low-resolution log spectrum bins, spectral deltas, and local context values.
    This keeps the project debuggable before moving to Mel/MFCC features.
    """
    if feature_dim != 70:
        raise ValueError("This educational extractor currently supports feature_dim=70")

    frames = frame_signal(waveform, sample_rate, fps, frame_ms)
    window = np.hanning(len(frames[0])).astype(np.float32)
    features = []
    previous_spec = np.zeros(20, dtype=np.float32)

    for frame in frames:
        abs_frame = np.abs(frame)
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        stats = np.asarray(
            [
                rms,
                float(np.mean(abs_frame)),
                float(np.max(abs_frame)),
                float(np.std(frame)),
                float(np.percentile(abs_frame, 25)),
                float(np.percentile(abs_frame, 50)),
                float(np.percentile(abs_frame, 75)),
                float(np.mean(frame > 0.0)),
                float(np.mean(np.diff(np.signbit(frame)) != 0)),
                float(np.mean(frame)),
            ],
            dtype=np.float32,
        )

        spectrum = np.abs(np.fft.rfft(frame * window))
        spectrum = np.log1p(spectrum).astype(np.float32)
        chunks = np.array_split(spectrum, 20)
        spec20 = np.asarray([float(np.mean(chunk)) for chunk in chunks], dtype=np.float32)
        spec20 = spec20 / (float(np.max(spec20)) + 1e-6)
        delta20 = spec20 - previous_spec
        previous_spec = spec20
        context20 = np.clip(spec20 * (0.5 + min(rms / 0.2, 1.0) * 0.5), 0.0, 1.0)
        features.append(np.concatenate([stats, spec20, delta20, context20], axis=0))

    return np.stack(features, axis=0).astype(np.float32)


def smooth_params(params: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    params = np.asarray(params, dtype=np.float32)
    out = np.zeros_like(params)
    prev = np.zeros(params.shape[1:], dtype=np.float32)
    for idx, value in enumerate(params):
        prev = alpha * prev + (1.0 - alpha) * value
        out[idx] = prev
    return out


def frame_times(num_frames: int, fps: int) -> np.ndarray:
    return np.arange(num_frames, dtype=np.float32) / float(fps)
