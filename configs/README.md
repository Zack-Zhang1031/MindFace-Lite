# 配置目录

配置按功能域组织：

- `demos/`：规则、Stage 1.5、Stage 1.6 视觉演示。
- `datasets/`：合成数据、GRID 音频和 landmark 数据准备。
- `training/`：MLP、LSTM、TCN、Transformer 训练。
- `inference/`：PyTorch 和 ONNXRuntime 推理。
- `optimization/`：ONNX 量化、模型剪枝和 fine-tune。
- `benchmarks/`：运行时、量化、剪枝和后端一致性比较。
- `realtime/`：队列、TTS 和麦克风输入。
- `deployment/`：ONNX 导出、RKNN、Device Tree 和 U-Boot。

新命令应使用分组后的 kebab-case 路径。历史命令中的 `configs/*.yaml` 旧路径由 `mindface.utils.config` 自动映射。
