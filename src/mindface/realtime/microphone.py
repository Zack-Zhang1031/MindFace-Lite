from __future__ import annotations

import csv
import socket
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from time import perf_counter, sleep

import cv2
import numpy as np

from mindface.utils.config import resolve_path
from mindface.visual.mouth_drawer import draw_face_frame


@dataclass(frozen=True)
class MicFrame:
    frame_index: int
    time_sec: float
    rms: float
    mouth_open: float
    latency_ms: float


def check_sounddevice_available() -> tuple[bool, str]:
    try:
        import sounddevice as sd
    except ImportError:
        return (
            False,
            "sounddevice is not installed. Install it in the Windows training env "
            "or a separate optional-feature WSL env with: python -m pip install sounddevice",
        )
    except OSError as exc:
        return (
            False,
            "sounddevice is installed, but the PortAudio system library is unavailable. "
            f"Original error: {exc}. On Ubuntu/WSL install with: "
            "sudo apt install -y portaudio19-dev libportaudio2",
        )
    input_devices = get_input_devices(sd)
    if not input_devices:
        return (
            False,
            "sounddevice is installed, but PortAudio cannot see any input microphone device. "
            f"Default device={getattr(sd.default, 'device', None)}. "
            "Run: mindface realtime microphone --list-devices. "
            "Recommended fix: run microphone streaming in the Windows mindface-lite env. "
            "If you must use WSL, make sure Windows microphone permission is enabled and WSL audio input is exposed.",
        )
    return True, "sounddevice is installed."


def get_input_devices(sd_module=None) -> list[tuple[int, dict]]:
    if sd_module is None:
        import sounddevice as sd_module
    try:
        devices = sd_module.query_devices()
    except Exception:
        return []
    input_devices: list[tuple[int, dict]] = []
    for index, info in enumerate(devices):
        try:
            max_input_channels = int(info.get("max_input_channels", 0))
        except AttributeError:
            max_input_channels = int(info["max_input_channels"])
        if max_input_channels > 0:
            input_devices.append((index, dict(info)))
    return input_devices


def format_audio_device_report() -> str:
    try:
        import sounddevice as sd
    except ImportError:
        return "sounddevice is not installed. Run: python -m pip install sounddevice"
    except OSError as exc:
        return (
            "sounddevice is installed, but PortAudio is unavailable. "
            f"Original error: {exc}. On Ubuntu/WSL install: "
            "sudo apt install -y portaudio19-dev libportaudio2"
        )

    lines = [f"default.device={sd.default.device}"]
    try:
        devices = sd.query_devices()
    except Exception as exc:
        return "\n".join(lines + [f"query_devices failed: {exc}"])

    if len(devices) == 0:
        lines.append("No PortAudio devices found.")
        return "\n".join(lines)

    for index, info in enumerate(devices):
        marker = "input" if int(info.get("max_input_channels", 0)) > 0 else "output"
        lines.append(
            f"[{index}] {marker} name={info.get('name')} "
            f"max_input_channels={info.get('max_input_channels')} "
            f"default_samplerate={info.get('default_samplerate')}"
        )
    return "\n".join(lines)


def _make_device_error_message(device: int | str | None, exc: Exception) -> str:
    return (
        "Failed to open microphone input stream. "
        f"Configured device={device!r}. Original error: {exc}\n"
        + format_audio_device_report()
        + "\nIf default.device is [-1, -1] or no input devices are listed, "
        "PortAudio cannot see a microphone. Run this demo in the Windows mindface-lite env, "
        "or configure WSL audio input and set audio.device to a valid input device index/name."
    )


def _map_rms_to_open(rms: float, noise_floor: float, scale: float, gamma: float) -> float:
    value = np.clip((float(rms) - noise_floor) / max(scale - noise_floor, 1e-6), 0.0, 1.0)
    return float(np.power(value, gamma))


def run_microphone_rule_stream(
    sample_rate: int,
    fps: int,
    duration_sec: float,
    channels: int,
    device: int | str | None,
    noise_floor: float,
    scale: float,
    gamma: float,
    smoothing: float,
    csv_path: str | Path,
    show_window: bool,
    udp_host: str | None,
    udp_port: int | None,
    logger,
) -> list[MicFrame]:
    ok, message = check_sounddevice_available()
    if not ok:
        raise RuntimeError(message)
    import sounddevice as sd

    blocksize = max(1, int(sample_rate / fps))
    audio_queue: Queue[tuple[int, float, np.ndarray]] = Queue(maxsize=16)
    rows: list[MicFrame] = []
    dropped = 0

    udp_socket = None
    if udp_host and udp_port:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def callback(indata, frames, time_info, status) -> None:
        nonlocal dropped
        if status:
            logger.warning("sounddevice status: %s", status)
        audio = np.asarray(indata, dtype=np.float32).copy()
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        try:
            audio_queue.put_nowait((len(rows) + audio_queue.qsize(), perf_counter(), audio))
        except Full:
            dropped += 1

    csv_path = resolve_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    smoothed = 0.0
    frame_index = 0

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=blocksize,
            dtype="float32",
            device=device,
            callback=callback,
        ):
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["frame_index", "time_sec", "rms", "mouth_open", "latency_ms"])
                writer.writeheader()
                while perf_counter() - start_time < duration_sec:
                    try:
                        _, created_at, audio = audio_queue.get(timeout=0.25)
                    except Empty:
                        sleep(0.001)
                        continue
                    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
                    current = _map_rms_to_open(rms, noise_floor=noise_floor, scale=scale, gamma=gamma)
                    smoothed = smoothing * smoothed + (1.0 - smoothing) * current
                    latency_ms = (perf_counter() - created_at) * 1000.0
                    time_sec = frame_index / float(fps)
                    item = MicFrame(frame_index, time_sec, rms, float(smoothed), latency_ms)
                    rows.append(item)
                    writer.writerow(
                        {
                            "frame_index": item.frame_index,
                            "time_sec": f"{item.time_sec:.6f}",
                            "rms": f"{item.rms:.8f}",
                            "mouth_open": f"{item.mouth_open:.6f}",
                            "latency_ms": f"{item.latency_ms:.3f}",
                        }
                    )
                    if udp_socket is not None and udp_host and udp_port:
                        payload = f"{item.frame_index},{item.time_sec:.6f},{item.mouth_open:.6f}".encode("utf-8")
                        udp_socket.sendto(payload, (udp_host, udp_port))
                    if show_window:
                        frame = draw_face_frame(item.mouth_open)
                        cv2.imshow("MindFace-Lite Mic Rule Demo", frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    frame_index += 1
    except Exception as exc:
        if exc.__class__.__name__ == "PortAudioError":
            raise RuntimeError(_make_device_error_message(device, exc)) from exc
        raise

    if show_window:
        try:
            cv2.destroyWindow("MindFace-Lite Mic Rule Demo")
        except cv2.error:
            logger.warning("Failed to close OpenCV preview window cleanly.")
    if udp_socket is not None:
        udp_socket.close()
    logger.info("Microphone stream complete frames=%d dropped=%d csv=%s", len(rows), dropped, csv_path)
    return rows
