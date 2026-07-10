from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.realtime.pipeline import run_rule_queue_pipeline
from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run queue-based realtime rule simulation.")
    parser.add_argument("--config", default="configs/realtime_rule.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("realtime_rule", cfg["logging"]["log_path"])
    rows, stats = run_rule_queue_pipeline(
        str(resolve_path(cfg["audio"]["input_path"])),
        fps=int(cfg["audio"]["fps"]),
        frame_ms=float(cfg["audio"]["frame_ms"]),
        realtime_sleep=bool(cfg["realtime"]["sleep_like_realtime"]),
    )

    csv_path = resolve_path(cfg["output"]["csv_path"])
    report_path = resolve_path(cfg["output"]["report_path"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "time_sec", "rms", "mouth_open", "latency_ms"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_index": row.index,
                    "time_sec": f"{row.time_sec:.6f}",
                    "rms": f"{row.rms:.8f}",
                    "mouth_open": f"{row.mouth_open:.6f}",
                    "latency_ms": f"{row.latency_ms:.4f}",
                }
            )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"frames": len(rows), "latency": stats}, f, indent=2)

    logger.info("Realtime rule simulation frames=%d", len(rows))
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
