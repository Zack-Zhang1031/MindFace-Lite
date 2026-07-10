from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "outputs/audio/test_voice.wav",
    "outputs/videos/rule_mouth_demo.mp4",
    "outputs/videos/better_visual_mouth_demo.mp4",
    "outputs/videos/expressive_avatar_demo.mp4",
    "outputs/videos/expressive_avatar_preview.png",
    "outputs/logs/rule_demo.csv",
    "outputs/logs/better_visual_rule_demo.csv",
    "outputs/logs/expressive_avatar_demo.csv",
    "data/synthetic_mouth/manifest.csv",
    "outputs/checkpoints/mlp_mouth.pt",
    "outputs/videos/pytorch_mlp_demo.mp4",
    "outputs/models/mlp_mouth.onnx",
    "outputs/videos/onnx_mlp_demo.mp4",
    "outputs/reports/benchmark_report.json",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        print("Missing outputs:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)
    print("All basic outputs exist.")


if __name__ == "__main__":
    main()
