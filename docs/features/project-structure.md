# 项目结构整理

## overview

项目将配置、外部模型、生成模型和脚本职责明确分开，减少平铺文件数量和安装生成物造成的 Git 噪音。

## design decisions

- `configs/` 按功能域分为 demos、datasets、training、inference、optimization、benchmarks、realtime、deployment。
- 新配置文件使用 kebab-case；`src/mindface` 中的 Python 模块继续使用可导入的 snake_case。
- 旧 `configs/*.yaml` 路径由 `mindface.utils.config` 自动映射，已有命令仍可运行。
- `models/external/` 只存放第三方模型，例如 MediaPipe FaceLandmarker。
- `outputs/checkpoints/`、`outputs/models/` 只存放训练或转换生成的 PyTorch、ONNX、RKNN 产物。
- `*.egg-info/` 是 editable install 生成物，必须忽略且不提交。
- 编号脚本负责参数解析，复用逻辑放在 `src/mindface` 中。

## implementation notes

配置目录：

```text
configs/
├── demos/
├── datasets/
├── training/
├── inference/
├── optimization/
├── benchmarks/
├── realtime/
└── deployment/
```

模型目录：

```text
models/external/mediapipe/face_landmarker.task
outputs/checkpoints/*.pt
outputs/models/*.onnx
outputs/models/*.rknn
```

旧路径兼容示例：`configs/train_mlp.yaml` 会解析到 `configs/training/train-mlp.yaml`。新文档和新命令统一使用新路径。
