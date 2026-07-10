from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.tts.real_tts import check_tts_backend, generate_real_tts_wav
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WAV audio with a real TTS backend.")
    parser.add_argument("--config", default="configs/real_tts.yaml")
    parser.add_argument("--text", default=None)
    parser.add_argument("--engine", choices=["pyttsx3", "edge_tts"], default=None)
    parser.add_argument("--check-deps", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.engine is not None:
        cfg["engine"] = args.engine
    ok, message = check_tts_backend(str(cfg["engine"]))
    if args.check_deps:
        print(message)
        return
    if not ok:
        raise RuntimeError(message)

    logger = setup_logger("real_tts", cfg["logging"]["log_path"])
    text = args.text if args.text is not None else str(cfg["text"])
    output_path = generate_real_tts_wav(text, cfg)
    logger.info("Generated real TTS audio with engine=%s: %s", cfg["engine"], output_path)
    print(f"Real TTS WAV: {output_path}")


if __name__ == "__main__":
    main()
