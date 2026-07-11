from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str], optional: bool = False) -> None:
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=not optional)
    if optional and result.returncode != 0:
        print(f"Optional step failed with code {result.returncode}: {' '.join(command)}", flush=True)


def require_paths(paths: list[str]) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files for this optional pipeline:\n"
            + "\n".join(f"- {path}" for path in missing)
            + "\nRun the GRID preprocessing/training commands in RUN_ORDER.md first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MindFace-Lite staged validation pipeline.")
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip ONNX INT8 dynamic quantization and quantization benchmark.",
    )
    parser.add_argument(
        "--include-grid-compression",
        action="store_true",
        help="Also run GRID ONNX export/quantization and pruning benchmark. Requires GRID processed data and checkpoint.",
    )
    parser.add_argument(
        "--check-optional-deps",
        action="store_true",
        help="Check optional dependencies for landmark/TTS/mic/RKNN/Device Tree scripts.",
    )
    parser.add_argument(
        "--check-optional-deps-only",
        action="store_true",
        help="Only check optional dependencies, without running the basic pipeline.",
    )
    args = parser.parse_args()

    python = sys.executable
    optional_steps = [
        [python, "scripts/14_extract_grid_video_landmarks.py", "--check-deps"],
        [python, "scripts/21_real_tts_generate_wav.py", "--check-deps"],
        [python, "scripts/23_mic_stream_rule_demo.py", "--check-deps"],
        [python, "scripts/24_rknn_convert_and_infer.py", "--check-deps"],
        [python, "scripts/25_device_tree_uboot_check.py", "--check-deps"],
    ]
    if args.check_optional_deps_only:
        for step in optional_steps:
            run_step(step, optional=True)
        print("MindFace-Lite optional dependency check completed.")
        return

    steps = [
        [python, "scripts/00_generate_test_audio.py"],
        [python, "scripts/01_rule_mouth_demo.py", "--config", "configs/rule_demo.yaml"],
        [python, "scripts/01_5_better_visual_demo.py", "--config", "configs/better_visual_demo.yaml"],
        [python, "scripts/01_6_expressive_avatar_demo.py", "--config", "configs/expressive_avatar_demo.yaml"],
        [python, "scripts/02_generate_synthetic_dataset.py", "--config", "configs/synthetic_dataset.yaml"],
        [python, "scripts/03_train_model.py", "--config", "configs/train_mlp.yaml"],
        [python, "scripts/04_infer_pytorch.py", "--config", "configs/infer_pytorch.yaml"],
        [python, "scripts/05_export_onnx.py", "--config", "configs/export_onnx.yaml"],
        [python, "scripts/06_infer_onnx.py", "--config", "configs/infer_onnx.yaml"],
        [python, "scripts/17_compare_inference_backends.py", "--config", "configs/consistency_compare.yaml"],
        [python, "scripts/07_realtime_rule_demo.py", "--config", "configs/realtime_rule.yaml"],
        [python, "scripts/08_benchmark.py", "--config", "configs/benchmark.yaml"],
    ]
    if not args.skip_quantization:
        steps.extend(
            [
                [python, "scripts/10_quantize_onnx.py", "--config", "configs/quantize_onnx.yaml"],
                [python, "scripts/11_benchmark_quantized_onnx.py", "--config", "configs/benchmark_quantized_onnx.yaml"],
            ]
        )

    for step in steps:
        run_step(step)

    if args.include_grid_compression:
        require_paths(
            [
                "outputs/checkpoints/grid_mlp_mouth.pt",
                "data/processed/grid_mouth/manifest.csv",
            ]
        )
        grid_steps = [
            [python, "scripts/05_export_onnx.py", "--config", "configs/export_grid_onnx.yaml"],
            [python, "scripts/10_quantize_onnx.py", "--config", "configs/quantize_grid_onnx.yaml"],
            [python, "scripts/11_benchmark_quantized_onnx.py", "--config", "configs/benchmark_grid_quantized_onnx.yaml"],
            [python, "scripts/12_prune_finetune.py", "--config", "configs/prune_finetune.yaml"],
            [python, "scripts/13_benchmark_pruned.py", "--config", "configs/benchmark_pruned.yaml"],
        ]
        for step in grid_steps:
            run_step(step)

    if args.check_optional_deps:
        for step in optional_steps:
            run_step(step, optional=True)

    print("MindFace-Lite basic pipeline completed.")


if __name__ == "__main__":
    main()
