# Backend Consistency

## overview

Backend consistency compares the same audio feature input across PyTorch, ONNXRuntime, and optional RKNN inference. This is a deployment debugging step: before optimizing speed, first prove that different inference backends produce numerically similar mouth parameters.

## design decisions

- Use PyTorch output as the reference because the checkpoint is trained and saved in PyTorch.
- Compare ONNXRuntime on the same Windows training environment where ONNX export is produced.
- Treat RKNN as optional because real RKNN inference needs the WSL RKNN environment, RKNN runtime support, or an RK3588 board.
- Save a JSON report so benchmark and deployment notes can reference concrete numbers.

## implementation notes

Run after a checkpoint and ONNX model exist:

```powershell
python scripts/00_generate_test_audio.py
python scripts/03_train_model.py --config configs/train_mlp.yaml
python scripts/05_export_onnx.py --config configs/export_onnx.yaml
python scripts/17_compare_inference_backends.py --config configs/consistency_compare.yaml
```

Unified CLI:

```powershell
python -m mindface compare-backends
```

Report output:

```text
outputs/reports/backend_consistency_report.json
```

The report includes:

```text
pytorch.latency_ms
onnxruntime.latency_ms
onnxruntime.error_vs_pytorch.mae
onnxruntime.error_vs_pytorch.max_abs_error
onnxruntime.error_vs_pytorch.rmse
rknn.available
rknn.reason or rknn.error_vs_pytorch
```

Expected interpretation:

- PyTorch vs ONNXRuntime should be very close for the same MLP checkpoint, usually near floating point noise.
- RKNN may be unavailable on Windows; that is reported as `available: false`, not treated as a failed Windows training run.
- A large error means the input shape, model type metadata, exported ONNX graph, or backend preprocessing is inconsistent.
