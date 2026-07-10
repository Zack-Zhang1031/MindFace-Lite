# 性能报告

## 环境

- 日期：2026-07-08
- 机器：本地 Windows
- GPU：NVIDIA GeForce RTX 4080，用于训练和剪枝 fine-tune
- Runtime：CPU benchmark
- 模型：MLP mouth predictor
- 输入 shape：256 frames x 70 features

## 基础推理结果

来自 `outputs/reports/benchmark_report.json`：

```text
PyTorch 平均延迟: 约 0.30 ms / 256-frame batch
PyTorch 近似吞吐: 约 3338 FPS
ONNXRuntime 平均延迟: 约 3.05 ms / 256-frame batch
ONNXRuntime 近似吞吐: 约 327 FPS
```

## ONNX INT8 Dynamic Quantization

来自 `outputs/reports/quantized_onnx_benchmark.json`：

```text
FP32 ONNX 大小: 104888 bytes
INT8 dynamic ONNX 大小: 30980 bytes
压缩后大小比例: 0.295
输出 MAE: 0.003838
FP32 平均延迟: 3.665 ms
INT8 dynamic 平均延迟: 4.478 ms
```

GRID MLP 来自 `outputs/reports/grid_quantized_onnx_benchmark.json`：

```text
FP32 ONNX 大小: 104888 bytes
INT8 dynamic ONNX 大小: 30980 bytes
输出 MAE: 0.013362
FP32 平均延迟: 3.378 ms
INT8 dynamic 平均延迟: 4.254 ms
```

解释：动态量化明显减小模型大小，但这个模型太小，当前 CPU 上 INT8 不比 FP32 快。

## 剪枝

来自 `outputs/reports/pruned_benchmark.json`：

```text
原始模型稀疏率: 0
剪枝模型稀疏率: 0.299984
原始模型平均延迟: 0.305 ms
剪枝模型平均延迟: 0.322 ms
```

解释：L1 非结构化剪枝产生了稀疏权重，但普通 dense PyTorch 推理不会自动利用稀疏性，所以没有获得实际加速。

## 后续改进

- 测不同 batch size 和序列长度。
- 分离特征提取延迟和模型推理延迟。
- 在 streaming 模式下统计 P50、P95、P99。
- 比较 MLP、LSTM、TCN、Transformer checkpoint。
- 在 RK3588 CPU / NPU 上做 RKNN benchmark。
- 尝试结构化剪枝或目标 runtime 支持的稀疏加速。

## 面试解释

我能同时报告模型大小、输出误差、平均延迟、P95 延迟和吞吐。实时系统不能只看平均延迟，因为少数慢帧也会造成肉眼可见的嘴型抖动。
