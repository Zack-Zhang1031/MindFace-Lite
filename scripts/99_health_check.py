from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindface.diagnostics.health import run_health_checks
from mindface.utils.config import resolve_path


def print_report(report: dict) -> None:
    summary = report["summary"]
    print(
        "MindFace-Lite health check: "
        f"pass={summary.get('pass', 0)} warn={summary.get('warn', 0)} fail={summary.get('fail', 0)}"
    )
    for check in report["checks"]:
        status = str(check["status"]).upper()
        print(f"[{status:4}] {check['name']}: {check['message']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-command MindFace-Lite project health checks.")
    parser.add_argument("--output", default="outputs/reports/health_check.json")
    parser.add_argument("--json-only", action="store_true", help="Only write JSON, without the console table.")
    args = parser.parse_args()

    report = run_health_checks()
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if not args.json_only:
        print_report(report)
        print(f"Report: {output_path}")


if __name__ == "__main__":
    main()

