# Deployment Stage

## What To Build

Deployment converts a trained PyTorch model into forms that can run outside the training script:

- PyTorch checkpoint inference.
- ONNX export.
- ONNXRuntime inference.
- Performance benchmark.
- Quantization discussion for later edge deployment.

## Commands

```powershell
python scripts/04_infer_pytorch.py --config configs/infer_pytorch.yaml
python scripts/05_export_onnx.py --config configs/export_onnx.yaml
python scripts/06_infer_onnx.py --config configs/infer_onnx.yaml
python scripts/08_benchmark.py --config configs/benchmark.yaml
```

## Expected Outputs

```text
outputs/videos/pytorch_mlp_demo.mp4
outputs/logs/pytorch_mlp_params.csv
outputs/models/mlp_mouth.onnx
outputs/videos/onnx_mlp_demo.mp4
outputs/logs/onnx_mlp_params.csv
outputs/reports/benchmark_report.json
```

If export creates `mlp_mouth.onnx.data`, keep it next to the `.onnx` file when copying the model.

## Key Ideas

- PyTorch is convenient for training and research.
- ONNX is a portable model representation.
- ONNXRuntime is a production-style inference engine.
- Dynamic axes allow variable frame counts.
- Benchmarking reports latency and approximate FPS.

## Quantization Discussion

For edge deployment, INT8 quantization can reduce model size and improve throughput. It needs calibration data that represents real input feature distributions. Bad calibration can cause mouth motion to become flat, clipped, or unstable.

## Interview Explanation

I can explain the difference between training format and deployment format, export a model to ONNX, validate ONNXRuntime output, and measure latency. This is a practical algorithm-debugging skill because deployment bugs often happen at tensor shapes, normalization, dynamic axes, and unsupported operators.
