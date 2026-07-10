# 训练实验追踪

## overview

训练脚本会在每次运行时自动创建 `outputs/experiments/<timestamp>_train_<model>/`，保存本次训练的配置、epoch 曲线和最终指标。

## design decisions

- 不引入 MLflow/W&B，保持项目轻量。
- 不改变原有训练命令，追踪逻辑内置到 `train_from_config()`。
- 保存 config 快照，避免之后修改 YAML 后无法复现实验。

## implementation notes

每次训练会生成：

```text
outputs/experiments/<run>/
├── config.yaml
├── history.csv
└── metrics.json
outputs/experiments/latest_train_run.txt
```

`metrics.json` 包含模型类型、数据集路径、训练/验证样本数、最佳验证损失、参数量、设备、耗时、PyTorch/CUDA 信息。

