from __future__ import annotations

import numpy as np

from mindface.audio.features import write_wav_mono


def text_to_waveform(text: str, sample_rate: int = 16000, char_duration: float = 0.09) -> np.ndarray:
    """Generate speech-like test audio from text.

    This is not a real TTS model. It creates deterministic voiced pulses so the
    rest of the mouth pipeline can be tested before an external TTS is attached.
    """
    chunks: list[np.ndarray] = []
    for idx, char in enumerate(text):
        duration = char_duration * (1.8 if char.isspace() else 1.0)
        n = max(1, int(sample_rate * duration))
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        if char.isspace():
            chunks.append(np.zeros(n, dtype=np.float32))
            continue
        base = 150.0 + (ord(char) % 30) * 8.0
        envelope = np.clip(np.sin(np.linspace(0, np.pi, n, dtype=np.float32)), 0.0, 1.0) ** 0.7
        tone = np.sin(2 * np.pi * base * t) + 0.35 * np.sin(2 * np.pi * base * 2.0 * t)
        chunks.append((0.18 * envelope * tone).astype(np.float32))
        if idx % 5 == 4:
            chunks.append(np.zeros(int(sample_rate * 0.035), dtype=np.float32))
    if not chunks:
        return np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    return np.clip(np.concatenate(chunks), -1.0, 1.0).astype(np.float32)


def write_text_wav(text: str, output_path: str, sample_rate: int = 16000) -> None:
    write_wav_mono(output_path, sample_rate, text_to_waveform(text, sample_rate=sample_rate))
