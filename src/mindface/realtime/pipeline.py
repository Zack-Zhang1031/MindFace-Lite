from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Thread
from time import perf_counter, sleep

import numpy as np

from mindface.audio.features import compute_rms, read_wav_mono, rms_to_mouth_open
from mindface.realtime.bounded_queue import BoundedDropQueue, QueueStopped
from mindface.utils.stats import LatencyStats


@dataclass
class AudioFrame:
    index: int
    time_sec: float
    rms: float
    created_at: float


@dataclass
class MouthFrame:
    index: int
    time_sec: float
    rms: float
    mouth_open: float
    latency_ms: float


def run_rule_queue_pipeline(audio_path: str, fps: int = 25, frame_ms: float = 40.0, realtime_sleep: bool = False):
    """Simulate producer-consumer realtime mouth control from a WAV file."""
    sample_rate, waveform = read_wav_mono(audio_path)
    rms_values = compute_rms(waveform, sample_rate, fps=fps, frame_ms=frame_ms)
    mouth_values = rms_to_mouth_open(rms_values)
    in_queue = BoundedDropQueue[AudioFrame](maxsize=8, overflow="block")
    out_queue = BoundedDropQueue[MouthFrame](maxsize=8, overflow="block")
    stats = LatencyStats()

    def producer() -> None:
        try:
            for idx, rms in enumerate(rms_values):
                in_queue.put(AudioFrame(idx, idx / float(fps), float(rms), perf_counter()))
                if realtime_sleep:
                    sleep(1.0 / float(fps))
            in_queue.stop()
        except BaseException as exc:
            in_queue.fail(exc)

    def consumer() -> None:
        try:
            while True:
                try:
                    item = in_queue.get()
                except QueueStopped:
                    break
                start = perf_counter()
                mouth_open = float(mouth_values[item.index])
                latency_ms = (perf_counter() - item.created_at) * 1000.0
                stats.add_since(start)
                out_queue.put(MouthFrame(item.index, item.time_sec, item.rms, mouth_open, latency_ms))
            out_queue.stop()
        except BaseException as exc:
            out_queue.fail(exc)

    producer_thread = Thread(target=producer, name="audio-producer")
    consumer_thread = Thread(target=consumer, name="mouth-consumer")
    producer_thread.start()
    consumer_thread.start()

    rows: list[MouthFrame] = []
    try:
        while True:
            try:
                rows.append(out_queue.get())
            except QueueStopped:
                break
    finally:
        producer_thread.join()
        consumer_thread.join()

    summary = stats.summary()
    summary["input_queue"] = asdict(in_queue.stats())
    summary["output_queue"] = asdict(out_queue.stats())
    return rows, summary
