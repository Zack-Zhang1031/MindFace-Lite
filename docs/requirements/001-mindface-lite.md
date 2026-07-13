# MindFace-Lite 总需求

## Overview

MindFace-Lite 是一个面向 AI 算法工程 / AI 算法调试工程岗位的教育型项目。它从规则口型 demo 开始，逐步覆盖数据处理、PyTorch 训练、ONNX 部署、量化剪枝、实时输入、C++ 控制、RKNN/RK3588 边缘部署思路和文档交付。

## User Stories

- 作为学习者，我希望一条命令能检查当前环境和关键输出，快速知道项目是否能继续运行。
- 作为训练使用者，我希望每次训练自动保存配置、曲线和指标，方便复现实验。
- 作为数据处理使用者，我希望 GRID 视频 landmark 提取后得到质量报告，判断标签能否用于训练。
- 作为部署学习者，我希望 RKNN 转换后得到环境和转换报告，方便定位兼容问题。
- 作为面试候选人，我希望文档能按功能和需求组织，方便讲清楚工程闭环。
- 作为初学者，我希望使用方向键菜单浏览全部功能，并在执行前看到用途、默认命令和环境要求。
- 作为高级使用者，我希望常用操作直接使用预设，同时可以切换到自定义 YAML。

## Functional Reqs

- 支持 `mindface health` 生成项目健康检查报告。
- 支持 Stage 1.5 better visual renderer，使用同一套 RMS 规则输出更适合展示的 MP4。
- 支持 Stage 1.6 expressive static avatar，使用生成的静态人脸图和 OpenCV 嘴部 ROI 形变输出 MP4。
- 训练脚本必须自动写入实验目录，包括 config、history、metrics。
- GRID landmark 脚本必须能自动或单独生成质量报告。
- GRID landmark 标签必须能和 GRID WAV 音频特征对齐，生成可训练的 manifest。
- RKNN 转换脚本必须能保存部署报告 JSON。
- 必须提供 PyTorch、ONNXRuntime、RKNN 可选后端一致性对比报告。
- 必须提供统一 CLI，覆盖 Python、C++ 和项目检查入口，并按 pipeline、demo、data、infer、export、optimize、benchmark、realtime、tts、deploy、cpp、project 分组。
- 必须提供 `mindface ui` 方向键终端菜单，全部功能均可见，每项显示用途、命令和推荐环境。
- 配置型菜单项必须提供仓库预设，并允许输入自定义 YAML 路径。
- 编号脚本必须仅作为 CLI 兼容跳转入口，CLI 和 Pipeline 不得反向依赖编号脚本。
- 必须提供 pytest 自动测试入口，覆盖关键纯 Python 模块和轻量数据路径。
- 全部 YAML 必须有 schema 校验，错误必须包含字段路径。
- 训练产物必须使用版本化 ModelBundle，并保存统一 FeatureSpec、optimizer 和 epoch。
- 训练必须支持从 last checkpoint 恢复，且保留 best checkpoint 语义。
- PyTorch、ONNXRuntime、RKNN 必须实现统一 MouthPredictor 接口。
- Pipeline 必须支持 dry-run、阶段范围、已有产物跳过和 force 重跑。
- GRID 数据必须优先按 speaker-disjoint 切分并记录 manifest schema version。
- Python 实时队列必须覆盖满载、丢帧、停止和异常传播。
- C++ 必须拆分 runtime library、apps 和 CTest。
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
- ModelBundle：schema version、模型类型/参数、FeatureSpec、目标名称、权重、optimizer、epoch、最佳损失、训练配置、metadata。
- 数据 manifest：schema version、split strategy、speaker、特征/标签路径和来源追踪字段。

## UI/UX

本项目没有图形桌面 UI，但提供方向键终端工作台。主菜单按学习领域分组，使用 `↑/↓` 移动、Enter 确认、Esc/Q 返回。暂时不可运行的功能不会隐藏，而是显示 Windows CUDA、Windows 可选功能、WSL RKNN、Linux BSP 或真实硬件要求。

## API

- `mindface ui`
- `mindface health`
- `mindface demo better-visual --config configs/demos/better-visual-demo.yaml`
- `mindface demo expressive-avatar --config configs/demos/expressive-avatar-demo.yaml`
- `mindface train --config configs/training/train-mlp.yaml`
- `mindface data align-landmarks --config configs/datasets/prepare-grid-landmark.yaml`
- `mindface benchmark backends --config configs/benchmarks/backend-consistency.yaml`
- `mindface data extract-landmarks --config configs/datasets/grid-video-landmarks.yaml --quality-only`
- `mindface deploy rknn --config configs/deployment/rknn-deploy.yaml --dry-run`
- `mindface config validate --all`
- `mindface pipeline basic --from-step train --to-step benchmark --force`
- `mindface cpp configure`
- `mindface cpp build`
- `mindface cpp test`
- `mindface project test`

## Testing

- 使用 `mindface project compile` 检查脚本和模块语法。
- 使用 `mindface health` 验证健康检查报告能生成。
- 使用 CLI 路由测试递归覆盖全部 argparse 叶子命令。
- 使用 TUI 目录测试验证全部菜单动作都能被 CLI 解析，并且每项包含环境说明。
- 使用兼容脚本测试保证编号脚本没有 argparse 和业务逻辑。
- 使用 `mindface demo better-visual` 和 `mindface demo expressive-avatar` 验证视觉视频生成。
- 使用 `mindface data extract-landmarks --quality-only` 验证已有 landmark 输出能生成质量报告。
- 使用 `mindface data align-landmarks --max-samples 8` 验证真实 GRID landmark 监督训练集准备入口。
- 使用 `mindface benchmark backends` 验证 PyTorch/ONNXRuntime/RKNN 可选后端一致性报告。
- 使用 `python -m pytest` 验证自动测试。
- 使用 CMake build 和 CTest 验证 C++ runtime 与有界队列。
- 使用迷你 CPU 数据集验证训练、保存、加载、optimizer 恢复和 epoch 继续。
- 使用 `mindface deploy rknn --dry-run` 验证 RKNN 报告能在无板端运行时生成。

## Open Questions

- 是否要把健康检查纳入 `run_00_basic_pipeline.py` 的默认流程。
- 是否要把实验追踪进一步升级为 MLflow/W&B，这目前不建议，避免项目过重。
- 是否要增加真实 phoneme/viseme forced-alignment 工具，用音素级标签替代当前演示级 viseme 配置。
