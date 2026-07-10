# MindFace-Lite 总需求

## Overview

MindFace-Lite 是一个面向 AI 算法工程 / AI 算法调试工程岗位的教育型项目。它从规则口型 demo 开始，逐步覆盖数据处理、PyTorch 训练、ONNX 部署、量化剪枝、实时输入、C++ 控制、RKNN/RK3588 边缘部署思路和文档交付。

## User Stories

- 作为学习者，我希望一条命令能检查当前环境和关键输出，快速知道项目是否能继续运行。
- 作为训练使用者，我希望每次训练自动保存配置、曲线和指标，方便复现实验。
- 作为数据处理使用者，我希望 GRID 视频 landmark 提取后得到质量报告，判断标签能否用于训练。
- 作为部署学习者，我希望 RKNN 转换后得到环境和转换报告，方便定位兼容问题。
- 作为面试候选人，我希望文档能按功能和需求组织，方便讲清楚工程闭环。

## Functional Reqs

- 支持 `scripts/99_health_check.py` 生成项目健康检查报告。
- 支持 Stage 1.5 better visual renderer，使用同一套 RMS 规则输出更适合展示的 MP4。
- 支持 Stage 1.6 expressive static avatar，使用生成的静态人脸图和 OpenCV 嘴部 ROI 形变输出 MP4。
- 训练脚本必须自动写入实验目录，包括 config、history、metrics。
- GRID landmark 脚本必须能自动或单独生成质量报告。
- RKNN 转换脚本必须能保存部署报告 JSON。
- README 必须明确 Windows 训练环境、WSL RKNN 环境和硬件相关脚本的边界。
- `docs/requirements/` 和 `docs/features/` 必须有 README 索引。

## Non-Functional Reqs

- 所有脚本必须能从项目根目录运行。
- 重要输出路径必须可配置，默认落到 `outputs/`。
- 新增功能应保持轻量，不引入额外重依赖。
- 错误信息应能帮助定位环境、依赖、路径或数据问题。

## Data Model

- 健康检查报告：`summary`、`checks[]`、每个检查项包含 `name/status/message/details`。
- 实验追踪：`config.yaml`、`history.csv`、`metrics.json`、`latest_train_run.txt`。
- GRID 质量报告：样本数、帧数、检测率、口型参数分布、低检测样本列表。
- RKNN 部署报告：环境版本、ONNX 输入、RKNN 输出、量化数据集、转换状态、后续板端步骤。

## UI/UX

本项目没有图形 UI。用户体验重点是命令行可运行、输出路径明确、报告可读、失败时提示具体修复方向。

## API

- `python scripts/99_health_check.py`
- `python scripts/01_5_better_visual_demo.py --config configs/better_visual_demo.yaml`
- `python scripts/01_6_expressive_avatar_demo.py --config configs/expressive_avatar_demo.yaml`
- `python scripts/03_train_model.py --config configs/train_mlp.yaml`
- `python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --quality-only`
- `python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml --dry-run`

## Testing

- 使用 `python -m py_compile` 检查新增脚本和模块语法。
- 使用 `python scripts/99_health_check.py` 验证健康检查报告能生成。
- 使用 `python scripts/01_5_better_visual_demo.py --config configs/better_visual_demo.yaml` 验证 Stage 1.5 视频能生成。
- 使用 `python scripts/01_6_expressive_avatar_demo.py --config configs/expressive_avatar_demo.yaml` 验证 Stage 1.6 静态头像形变视频能生成。
- 使用 `python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --quality-only` 验证已有 landmark 输出能生成质量报告。
- 使用 `python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml --dry-run` 验证 RKNN 报告能在无板端运行时生成。

## Open Questions

- 是否要把健康检查纳入 `run_00_basic_pipeline.py` 的默认流程。
- 是否要增加 pytest，用自动测试覆盖 RMS、Dataset、ONNX 输出一致性和报告生成。
- 是否要把实验追踪进一步升级为 MLflow/W&B，这目前不建议，避免项目过重。
