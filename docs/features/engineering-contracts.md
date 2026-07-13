# 工程契约与可恢复运行

## overview

本功能把训练、推理、配置、Pipeline 和实时队列从松散约定升级为可验证契约，让错误在启动阶段暴露，并让训练产物能够跨 PyTorch、ONNXRuntime 和 RKNN 后端稳定复用。

## design decisions

- 所有 YAML 按目录和文件名验证必需字段、数值范围与切分比例。
- `FeatureSpec` 是 fps、frame_ms、feature_dim 的唯一来源，并保存在模型包中。
- `ModelBundle` 使用 schema version 1，兼容旧 checkpoint。
- best checkpoint 保存最佳模型，last checkpoint 每个 epoch 保存模型、optimizer 和 epoch。
- `MouthPredictor` 统一 PyTorch、ONNXRuntime 和 RKNN 的 `predict(features)`。
- Pipeline 用步骤声明命令和产物，支持 dry-run、范围选择、跳过和强制执行。
- Python/C++ 队列显式定义容量、丢帧、stop 和错误传播。
- GRID 音频优先 speaker-disjoint；无法识别视频 speaker 时记录 sample fallback。

## implementation notes

```powershell
python -m mindface config list
python -m mindface config show configs/training/train-mlp.yaml
python -m mindface config validate --all
python -m mindface pipeline basic --dry-run
python -m mindface pipeline basic --from-step train --to-step benchmark
python -m mindface pipeline basic --from-step train --to-step benchmark --force
```

恢复训练时，把 `train.resume_from` 指向 `output.last_checkpoint_path`，并把 `train.epochs` 设置为新的总 epoch 数。模型包包含模型类型、参数、FeatureSpec、目标名称、权重、optimizer、epoch、最佳验证损失、训练配置和 metadata。

C++ 使用 `cpp/CMakePresets.json`；公共运行库位于 `cpp/include/mindface` 和 `cpp/src`，应用位于 `cpp/apps`，CTest 位于 `cpp/tests`。
