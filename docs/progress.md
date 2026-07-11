# 项目进度

## 已完成

- 已创建 `src/mindface` 主包。
- 已实现 RMS 规则口型 demo。
- 已实现 Stage 1.5 better visual renderer，生成更精致的 OpenCV 2D 数字脸口型视频。
- 已实现 Stage 1.6 expressive static avatar，使用生成的静态人脸图和 OpenCV 嘴部 ROI 形变生成口型视频。
- 已实现合成数据集生成。
- 已实现 MLP、LSTM、TCN、Transformer 模型定义。
- 已实现通用 PyTorch 训练闭环。
- 已验证 MLP、LSTM、TCN、Transformer 的短训练路径。
- 已实现 PyTorch 推理、ONNX 导出、ONNXRuntime 推理和 benchmark。
- 已实现实时队列模拟。
- 已实现 C++ 队列、UDP、串口式输出和 ONNXRuntime C++ demo 结构。
- 已归档旧包、旧配置、旧脚本、旧文档到 `archive/legacy-2026-07-06/`。
- 已实现 GRID 预处理脚本，生成 `data/processed/grid_mouth/manifest.csv`。
- 已完成 GRID 全量音频伪标签训练，产物为 `outputs/checkpoints/grid_mlp_mouth.pt`。
- 已实现 ONNX INT8 dynamic quantization，并生成 `outputs/models/mlp_mouth.int8.dynamic.onnx`。
- 已实现量化前后 benchmark，报告为 `outputs/reports/quantized_onnx_benchmark.json`。
- 已导出并量化 GRID MLP ONNX，报告为 `outputs/reports/grid_quantized_onnx_benchmark.json`。
- 已实现 L1 非结构化剪枝、剪枝后 fine-tune 和 benchmark。
- 已生成剪枝模型 `outputs/checkpoints/grid_mlp_mouth.pruned.pt`。
- 已实现 GRID 视频 landmark 标签提取入口，依赖 `mediapipe`。
- 已实现 GRID landmark 监督训练数据准备入口 `scripts/16_prepare_grid_landmark_dataset.py`。
- 已新增 GRID landmark MLP 训练配置 `configs/train_grid_landmark_mlp.yaml`。
- 已实现真实 TTS 接入口，支持 `pyttsx3` 和 `edge_tts`。
- 已实现麦克风 streaming RMS 口型入口，依赖 `sounddevice`。
- 已实现 RKNN 转换/推理入口，依赖 RKNN 开发环境。
- 已实现 Device Tree overlay 编译检查和 U-Boot 操作提示脚本。
- 已实现一键健康检查 `scripts/99_health_check.py`，输出 `outputs/reports/health_check.json`。
- 已实现训练实验追踪，自动保存 `config.yaml`、`history.csv`、`metrics.json`。
- 已实现 GRID landmark 标签质量报告，输出 `outputs/reports/grid_video_landmark_quality.json`。
- 已升级 RKNN 部署报告，输出 `outputs/reports/rknn_deploy_report.json`。
- 已实现 PyTorch、ONNXRuntime、RKNN 可选后端一致性对比，输出 `outputs/reports/backend_consistency_report.json`。
- 已新增统一 CLI：`python -m mindface ...`。
- 已新增 pytest 测试入口，覆盖音频特征、数据集、GRID landmark 预处理、视觉渲染和后端误差计算。
- 已新增 MIT `LICENSE`。
- 已新增 GitHub Actions CI：只运行 `python -m compileall -q src scripts tests` 和 `python -m pytest`。
- 已将 README 预览图放入 `docs/media/mindface-lite-preview.png`，避免引用 `outputs/` 生成物。
- 已新增 `docs/grid-landmark-training-report.md`，用于数据恢复后记录真实 landmark 训练结果。
- 已按 AGENTS.md 文档规范新增 `docs/requirements/` 和 `docs/features/` 索引。

## 下一步真实改进

- 恢复或复制真实 `data/raw/grid` 后，跑 GRID 视频 landmark 全量标签。
- 用 `data/processed/grid_video_landmarks` 和 `data/raw/grid/audio` 生成 `data/processed/grid_landmark_mouth/manifest.csv`，再训练真正的视频口型监督模型。
- 在真实麦克风和真实声卡上测试 streaming 延迟。
- 在 RK3588/RKNN 环境中验证 ONNX 到 RKNN 的真实转换和板端推理。
- 和硬件/BSP 团队确认 UART/I2S/GPIO 管脚、Device Tree、U-Boot、内核设备节点。
- 继续扩展 pytest，覆盖健康检查、实验追踪、质量报告和 RKNN dry-run 报告。
