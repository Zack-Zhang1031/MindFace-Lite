# 模型目录

`models/external/` 只存放下载的第三方模型，例如：

```text
models/external/mediapipe/face_landmarker.task
```

训练和转换生成的模型不放在这里：

```text
outputs/checkpoints/   # PyTorch checkpoint
outputs/models/        # ONNX、量化 ONNX、RKNN
```

第三方二进制和生成模型由 `.gitignore` 排除。仓库只保留下载方法、配置和可复现脚本。
