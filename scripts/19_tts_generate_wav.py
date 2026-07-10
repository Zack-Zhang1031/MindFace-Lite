from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.tts.simple_tts import write_text_wav
from mindface.utils.config import load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple TTS-like WAV.")
    parser.add_argument("--config", default="configs/tts_demo.yaml")
    parser.add_argument("--text", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    text = args.text if args.text is not None else str(cfg["text"])
    write_text_wav(text, cfg["audio"]["output_path"], sample_rate=int(cfg["audio"]["sample_rate"]))
    print(f"TTS-like WAV: {resolve_path(cfg['audio']['output_path'])}")


if __name__ == "__main__":
    main()
