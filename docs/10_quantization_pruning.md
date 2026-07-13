# 10. 量化与剪枝实操

本阶段把“只有文档说明”的模型压缩能力补成可运行代码：

- ONNX INT8 dynamic quantization
- 量化前后模型大小、输出误差、延迟 benchmark
- PyTorch L1 非结构化剪枝
- 剪枝后 fine-tune
- 剪枝前后稀疏率与延迟 benchmark

## 量化命令

```powershell
python scripts/10_quantize_onnx.py --config configs/optimization/quantize-onnx.yaml
python scripts/11_benchmark_quantized_onnx.py --config configs/benchmarks/benchmark-quantized-onnx.yaml
```

输出：

```text
outputs/models/mlp_mouth.int8.dynamic.onnx
outputs/reports/quantized_onnx_benchmark.json
```

如果要量化已经训练好的 GRID MLP：

```powershell
python scripts/05_export_onnx.py --config configs/deployment/export-grid-onnx.yaml
python scripts/10_quantize_onnx.py --config configs/optimization/quantize-grid-onnx.yaml
python scripts/11_benchmark_quantized_onnx.py --config configs/benchmarks/benchmark-grid-quantized-onnx.yaml
```

输出：

```text
outputs/models/grid_mlp_mouth.onnx
outputs/models/grid_mlp_mouth.int8.dynamic.onnx
outputs/reports/grid_quantized_onnx_benchmark.json
```

当前实测结果：

```text
FP32 ONNX 大小: 104888 bytes
INT8 dynamic ONNX 大小: 30980 bytes
压缩后大小比例: 0.295
输出 MAE: 0.003838
输出最大绝对误差: 0.053723
FP32 平均延迟: 3.665 ms
INT8 dynamic 平均延迟: 4.478 ms
```

GRID MLP 实测结果：

```text
FP32 ONNX 大小: 104888 bytes
INT8 dynamic ONNX 大小: 30980 bytes
压缩后大小比例: 0.295
输出 MAE: 0.013362
输出最大绝对误差: 0.080560
FP32 平均延迟: 3.378 ms
INT8 dynamic 平均延迟: 4.254 ms
```

结论：动态量化明显减小模型体积，但这个 MLP 很小，当前 CPU 上 INT8 并没有变快。面试时要说明：量化不是必然加速，是否加速取决于模型结构、算子、runtime、CPU 指令集和 batch/shape。

## 剪枝命令

```powershell
python scripts/12_prune_finetune.py --config configs/optimization/prune-finetune.yaml
python scripts/13_benchmark_pruned.py --config configs/benchmarks/benchmark-pruned.yaml
```

输出：

```text
outputs/checkpoints/grid_mlp_mouth.pruned.pt
outputs/reports/pruned_benchmark.json
```

当前实测结果：

```text
剪枝方法: L1 unstructured pruning
剪枝比例: 30%
剪枝后 fine-tune epoch: 2
fine-tune val_loss epoch 1: 0.001530
fine-tune val_loss epoch 2: 0.001520
最终稀疏率: 0.299984
原始模型平均延迟: 0.305 ms
剪枝模型平均延迟: 0.322 ms
```

结论：非结构化剪枝产生了稀疏权重，但普通 PyTorch dense kernel 不会自动利用这些 0 权重，所以延迟没有改善。要获得真实加速，通常需要结构化剪枝、稀疏 kernel、编译器支持或硬件支持。

## 关键代码

- `src/mindface/deploy/quantization.py`：ONNXRuntime 动态量化和 benchmark。
- `scripts/10_quantize_onnx.py`：生成 INT8 dynamic ONNX。
- `scripts/11_benchmark_quantized_onnx.py`：比较 FP32/INT8 大小、误差、延迟。
- `src/mindface/training/pruning.py`：L1 剪枝、mask 保留 fine-tune、稀疏率统计。
- `scripts/12_prune_finetune.py`：加载 checkpoint、剪枝、微调、保存。
- `scripts/13_benchmark_pruned.py`：比较原始 checkpoint 和剪枝 checkpoint。

## 常见错误

- `ONNX file not found`：先运行 `python scripts/05_export_onnx.py --config configs/deployment/export-onnx.yaml`。
- `onnxruntime quantization is unavailable`：安装 `onnxruntime`。
- 剪枝后稀疏率变低：fine-tune 时没有保留 pruning mask。当前代码会在 fine-tune 期间保留 mask，最后再固化权重。
- 剪枝不加速：非结构化稀疏不等于运行时加速，这是正常结果。

## 面试解释

我实现过 ONNX INT8 dynamic quantization，并用 benchmark 量化模型大小、输出误差和延迟。我也实现过 PyTorch L1 非结构化剪枝、剪枝后 fine-tune 和剪枝前后 benchmark。这个实验说明，模型压缩要同时看准确性、大小和真实 runtime 延迟，不能只看参数量或稀疏率。
