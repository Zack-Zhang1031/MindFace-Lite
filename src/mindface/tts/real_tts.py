from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mindface.utils.config import resolve_path


def check_tts_backend(engine: str) -> tuple[bool, str]:
    engine = engine.lower()
    if engine == "pyttsx3":
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            return False, "pyttsx3 is not installed. Run: python -m pip install pyttsx3"
        return True, "pyttsx3 is installed."
    if engine == "edge_tts":
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False, "edge_tts is not installed. Run: python -m pip install edge-tts"
        if shutil.which("ffmpeg") is None:
            return False, "ffmpeg is required to convert edge_tts MP3 output to WAV."
        return True, "edge_tts and ffmpeg are available."
    return False, f"Unsupported TTS engine={engine}. Use pyttsx3 or edge_tts."


def generate_pyttsx3_wav(
    text: str,
    output_path: str | Path,
    rate: int = 170,
    volume: float = 1.0,
    voice_contains: str | None = None,
    isolate_subprocess: bool | str = "auto",
) -> Path:
    output_path = resolve_path(output_path)
    should_isolate = _should_isolate_pyttsx3(isolate_subprocess)
    if should_isolate:
        return _generate_pyttsx3_wav_subprocess(
            text=text,
            output_path=output_path,
            rate=rate,
            volume=volume,
            voice_contains=voice_contains,
        )
    return _generate_pyttsx3_wav_direct(
        text=text,
        output_path=output_path,
        rate=rate,
        volume=volume,
        voice_contains=voice_contains,
    )


def _should_isolate_pyttsx3(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized == "auto":
        return platform.system() == "Linux"
    raise ValueError("pyttsx3 isolate_subprocess must be true, false, or auto.")


def _generate_pyttsx3_wav_direct(
    text: str,
    output_path: str | Path,
    rate: int = 170,
    volume: float = 1.0,
    voice_contains: str | None = None,
) -> Path:
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("pyttsx3 is not installed. Run: python -m pip install pyttsx3") from exc

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    engine.setProperty("rate", int(rate))
    engine.setProperty("volume", float(volume))
    if voice_contains:
        needle = voice_contains.lower()
        for voice in engine.getProperty("voices"):
            voice_text = f"{voice.id} {voice.name}".lower()
            if needle in voice_text:
                engine.setProperty("voice", voice.id)
                break
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()
    engine.stop()
    if not output_path.exists():
        raise RuntimeError(f"pyttsx3 did not create the expected WAV file: {output_path}")
    return output_path


def _generate_pyttsx3_wav_subprocess(
    text: str,
    output_path: Path,
    rate: int,
    volume: float,
    voice_contains: str | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    payload = {
        "text": text,
        "output_path": str(output_path),
        "rate": int(rate),
        "volume": float(volume),
        "voice_contains": voice_contains,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_path = Path(f.name)

    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_dir if not existing_pythonpath else src_dir + os.pathsep + existing_pythonpath

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "mindface.tts.real_tts", "--pyttsx3-worker", str(payload_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    finally:
        payload_path.unlink(missing_ok=True)

    if output_path.exists() and output_path.stat().st_size > 44:
        return output_path

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    details = "\n".join(part for part in (stdout, stderr) if part)
    raise RuntimeError(
        "pyttsx3 failed to create a valid WAV file. "
        f"Return code={completed.returncode}. Details: {details}"
    )


async def _edge_tts_to_mp3(text: str, voice: str, rate: str, volume: str, output_mp3: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    await communicate.save(str(output_mp3))


def generate_edge_tts_wav(
    text: str,
    output_path: str | Path,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    volume: str = "+0%",
    sample_rate: int = 16000,
) -> Path:
    try:
        import edge_tts  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("edge_tts is not installed. Run: python -m pip install edge-tts") from exc
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert edge_tts MP3 output to WAV.")

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        mp3_path = Path(tmp_dir) / "tts.mp3"
        asyncio.run(_edge_tts_to_mp3(text, voice, rate, volume, mp3_path))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp3_path),
                "-ar",
                str(int(sample_rate)),
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
        )
    return output_path


def generate_real_tts_wav(text: str, cfg: dict) -> Path:
    engine = str(cfg["engine"]).lower()
    output_path = cfg["audio"]["output_path"]
    if engine == "pyttsx3":
        return generate_pyttsx3_wav(
            text,
            output_path,
            rate=int(cfg["pyttsx3"].get("rate", 170)),
            volume=float(cfg["pyttsx3"].get("volume", 1.0)),
            voice_contains=cfg["pyttsx3"].get("voice_contains"),
            isolate_subprocess=cfg["pyttsx3"].get("isolate_subprocess", "auto"),
        )
    if engine == "edge_tts":
        return generate_edge_tts_wav(
            text,
            output_path,
            voice=str(cfg["edge_tts"].get("voice", "zh-CN-XiaoxiaoNeural")),
            rate=str(cfg["edge_tts"].get("rate", "+0%")),
            volume=str(cfg["edge_tts"].get("volume", "+0%")),
            sample_rate=int(cfg["audio"].get("sample_rate", 16000)),
        )
    raise ValueError(f"Unsupported TTS engine={engine}. Use pyttsx3 or edge_tts.")


def _run_pyttsx3_worker(payload_path: str | Path) -> None:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    _generate_pyttsx3_wav_direct(
        text=str(payload["text"]),
        output_path=payload["output_path"],
        rate=int(payload["rate"]),
        volume=float(payload["volume"]),
        voice_contains=payload.get("voice_contains"),
    )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyttsx3-worker")
    args = parser.parse_args()
    if args.pyttsx3_worker:
        _run_pyttsx3_worker(args.pyttsx3_worker)


if __name__ == "__main__":
    _main()
