from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.realtime.microphone import check_sounddevice_available, format_audio_device_report, run_microphone_rule_stream
from mindface.utils.config import load_yaml
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realtime microphone RMS mouth control.")
    parser.add_argument("--config", default="configs/mic_stream.yaml")
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview window.")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--list-devices", action="store_true", help="List PortAudio devices visible to sounddevice.")
    args = parser.parse_args()

    if args.list_devices:
        print(format_audio_device_report())
        return

    ok, message = check_sounddevice_available()
    if args.check_deps:
        print(message)
        return
    if not ok:
        raise SystemExit(message)

    cfg = load_yaml(args.config)
    logger = setup_logger("mic_stream_rule_demo", cfg["logging"]["log_path"])
    udp_host = str(cfg["udp"]["host"]) if bool(cfg["udp"].get("enabled", False)) else None
    udp_port = int(cfg["udp"]["port"]) if bool(cfg["udp"].get("enabled", False)) else None
    try:
        rows = run_microphone_rule_stream(
            sample_rate=int(cfg["audio"]["sample_rate"]),
            fps=int(cfg["audio"]["fps"]),
            duration_sec=float(args.duration_sec if args.duration_sec is not None else cfg["audio"]["duration_sec"]),
            channels=int(cfg["audio"].get("channels", 1)),
            device=cfg["audio"].get("device"),
            noise_floor=float(cfg["mouth"]["noise_floor"]),
            scale=float(cfg["mouth"]["scale"]),
            gamma=float(cfg["mouth"]["gamma"]),
            smoothing=float(cfg["mouth"]["smoothing"]),
            csv_path=cfg["output"]["csv_path"],
            show_window=bool(args.show or cfg["output"].get("show_window", False)),
            udp_host=udp_host,
            udp_port=udp_port,
            logger=logger,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Mic frames: {len(rows)}")
    print(f"CSV: {cfg['output']['csv_path']}")


if __name__ == "__main__":
    main()
