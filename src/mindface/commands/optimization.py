from __future__ import annotations

import argparse
import json

from mindface.utils.config import load_yaml, resolve_path
from mindface.utils.logger import setup_logger


def run_quantize(args: argparse.Namespace) -> int:
    from mindface.deploy.quantization import quantize_onnx_dynamic

    cfg = load_yaml(args.config)
    logger = setup_logger("quantize_onnx", cfg["logging"]["log_path"])
    output_path = quantize_onnx_dynamic(
        cfg["onnx"]["input_path"],
        cfg["onnx"]["output_path"],
        weight_type=str(cfg["quantization"]["weight_type"]),
        per_channel=bool(cfg["quantization"]["per_channel"]),
    )
    logger.info("Quantized ONNX model: %s", output_path)
    print(f"Quantized ONNX: {output_path}")
    return 0


def run_benchmark_quantized(args: argparse.Namespace) -> int:
    from mindface.deploy.quantization import benchmark_onnx_pair, write_json_report

    cfg = load_yaml(args.config)
    logger = setup_logger("benchmark_quantized_onnx", cfg["logging"]["log_path"])
    report = benchmark_onnx_pair(
        cfg["onnx"]["fp32_path"],
        cfg["onnx"]["int8_path"],
        model_type=str(cfg["onnx"]["model_type"]),
        frames=int(cfg["benchmark"]["frames"]),
        feature_dim=int(cfg["benchmark"]["feature_dim"]),
        warmup=int(cfg["benchmark"]["warmup"]),
        repeat=int(cfg["benchmark"]["repeat"]),
        seed=int(cfg["benchmark"]["seed"]),
    )
    report_path = write_json_report(cfg["output"]["report_path"], report)
    logger.info("Quantized ONNX benchmark report: %s", report_path)
    print(f"Report: {report_path}")
    return 0


def run_prune(args: argparse.Namespace) -> int:
    from mindface.training.pruning import prune_and_finetune_from_config

    cfg = load_yaml(args.config)
    logger = setup_logger("prune_finetune", cfg["logging"]["log_path"])
    output_path = prune_and_finetune_from_config(cfg, logger)
    print(f"Pruned checkpoint: {output_path}")
    return 0


def run_benchmark_pruned(args: argparse.Namespace) -> int:
    from mindface.training.pruning import benchmark_checkpoints

    cfg = load_yaml(args.config)
    logger = setup_logger("benchmark_pruned", cfg["logging"]["log_path"])
    report = benchmark_checkpoints(
        cfg["checkpoints"],
        frames=int(cfg["benchmark"]["frames"]),
        feature_dim=int(cfg["benchmark"]["feature_dim"]),
        warmup=int(cfg["benchmark"]["warmup"]),
        repeat=int(cfg["benchmark"]["repeat"]),
        seed=int(cfg["benchmark"]["seed"]),
    )
    output_path = resolve_path(cfg["output"]["report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Pruned benchmark report: %s", output_path)
    print(f"Report: {output_path}")
    return 0


def run_benchmark_runtime(args: argparse.Namespace) -> int:
    from mindface.deploy.benchmark import benchmark_from_config

    cfg = load_yaml(args.config)
    report_path = benchmark_from_config(cfg, setup_logger("benchmark", cfg["logging"]["log_path"]))
    print(f"Report: {report_path}")
    return 0


def run_benchmark_backends(args: argparse.Namespace) -> int:
    from mindface.deploy.consistency import compare_backends_from_config, write_consistency_report

    cfg = load_yaml(args.config)
    logger = setup_logger("consistency_compare", cfg["logging"]["log_path"])
    report = compare_backends_from_config(cfg)
    output_path = write_consistency_report(report, cfg["output"]["report_path"])
    logger.info("Consistency report: %s", output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {output_path}")
    return 0
