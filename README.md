# MindFace-Lite

![MindFace-Lite expressive avatar preview](docs/media/mindface-lite-preview.png)

MindFace-Lite 是一个面向 AI 算法工程 / AI 算法调试工程岗位的教育型工程项目。项目目标是从一个简单的“音量驱动嘴巴张合”规则 demo 出发，逐步扩展成可训练、可推理、可部署、可调试的轻量实时语音驱动数字人口型系统。

它不是一个追求生产级数字人效果的项目，而是一个完整工程学习闭环：数据处理、PyTorch 训练、ONNX 部署、量化剪枝、实时队列、C++ 控制、RK3588/RKNN 边缘部署思路，以及可讲清楚的调试文档。

## 项目能力

| 模块 | 当前能力 | 学习重点 |
| --- | --- | --- |
| 规则口型 demo | WAV 输入，计算 RMS，映射 `mouth_open`，生成 CSV 和 MP4；Stage 1.5 增加更精致 2D 数字脸渲染；Stage 1.6 使用静态人脸图做嘴部 ROI 形变 | 音频帧、RMS、规则系统、OpenCV 可视化 |
| PyTorch 训练 | 支持 MLP、LSTM、TCN、Transformer 配置切换 | Dataset、DataLoader、loss、optimizer、train/val/save/load |
| GRID 数据 | 支持 GRID 音频预处理、伪标签训练、视频 landmark 标签提取、landmark 监督数据集生成和训练配置 | 真实数据扫描、manifest、特征/标签落盘 |
| 推理部署 | PyTorch 推理、ONNX 导出、ONNXRuntime 推理、backend consistency、benchmark | 部署链路、性能统计、模型输入输出一致性 |
| 模型压缩 | ONNX INT8 dynamic quantization、剪枝、fine-tune、对比 benchmark | 压缩前后精度和速度评估 |
| TTS / 麦克风 | 伪 TTS、真实 TTS、麦克风 streaming RMS demo | 真实输入、实时 callback、延迟和 FPS |
| C++ 控制 | CMake、队列、多线程、UDP/串口输出 demo | 实时控制侧工程基础 |
| RKNN / RK3588 | ONNX -> RKNN 转换、DTBO 编译、U-Boot 操作提示 | 边缘部署、NPU、Device Tree、交叉编译 |

## 当前状态

已验证：

- 基础 pipeline 已跑通：规则 demo、Stage 1.5 better visual renderer、Stage 1.6 expressive static avatar、合成数据、MLP 训练、推理、ONNX、ONNXRuntime、benchmark。
- GRID 音频预处理和 GRID MLP 训练路径已实现。
- ONNX 动态量化、量化前后 benchmark、剪枝 fine-tune、剪枝 benchmark 已实现。
- GRID 视频 landmark 提取入口已实现，使用 MediaPipe Face Landmarker Tasks API。
- GRID landmark 监督训练入口已实现：`16_prepare_grid_landmark_dataset.py` + `train_grid_landmark_mlp.yaml`。
- 真实 TTS、麦克风 streaming、C++ demo、RKNN 转换、Device Tree overlay 编译入口已实现。
- 一键健康检查、训练实验追踪、GRID landmark 质量报告、RKNN 部署报告已实现。
- 统一 CLI、pytest 测试入口、PyTorch vs ONNXRuntime vs RKNN 可选一致性报告已实现。

需要注意：

- GRID 音频训练默认使用 RMS 规则伪标签，不等同于生产级 lip-sync 监督标签。
- 真正的视频口型监督训练需要 `data/raw/grid/audio`、`data/raw/grid/video` 和已提取的 `data/processed/grid_video_landmarks`。如果当前工作区没有 `data/raw/grid`，只能验证脚本入口，不能真实训练。
- MediaPipe landmark 在当前 Windows pip 包中走 CPU/XNNPACK；PyTorch 训练可以使用 NVIDIA GPU；RKNN 是后续面向 RK3588 NPU 的部署链路。
- `mindface-lite` 和 `mindface-rknn` 必须分环境使用，不能把所有依赖混在一个环境里。

## 环境选择

项目推荐两个环境，职责分开：

| 环境 | 平台 | 用途 |
| --- | --- | --- |
| `mindface-lite` | Windows conda | 训练、推理、ONNX、ONNXRuntime、量化、剪枝、MediaPipe、TTS、麦克风、C++ Windows demo |
| `mindface-rknn` | WSL/Ubuntu venv | RKNN 转换、Device Tree Compiler、ARM64 交叉编译 |

脚本环境矩阵：

| 脚本范围 | 推荐环境 | 说明 |
| --- | --- | --- |
| `00` 到 `13` | Windows `mindface-lite` | 基础 demo、训练、ONNX、量化、剪枝、benchmark |
| `14` | Windows `mindface-lite` | GRID 视频 landmark，MediaPipe Windows 环境已验证 |
| `19` 到 `23` | Windows `mindface-lite` | TTS、麦克风、OpenCV 窗口预览更稳定 |
| `24` | WSL `mindface-rknn` | RKNN-Toolkit2 转换环境 |
| `25` | WSL `mindface-rknn` | Device Tree Compiler 和 ARM64 交叉编译 |
| `cpp/` | Windows 或 WSL | Windows 侧生成 `.exe`，WSL 侧可做交叉编译练习 |

不要在 `mindface-rknn` 里安装 `requirements.txt` 或 `requirements-optional.txt`。原因是 `rknn-toolkit2==2.3.2` 明确要求 `numpy<=1.26.4`，而 Linux 上的新版 `mediapipe/opencv-contrib-python` 可能拉起 NumPy 2.x，导致 RKNN 环境损坏。

`requirements.txt` 和 `requirements-optional.txt` 也不是无脑追最新版本，而是有意把 NumPy 固定在 1.x，并把 OpenCV 固定在 `4.11.0.86`，避开 `opencv-contrib-python 5.x -> numpy 2.x` 的冲突。

## 安装

Windows 训练环境：

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
conda activate mindface-lite
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pip check
```

当前 `requirements.txt` 默认安装 CUDA 版 PyTorch：

```text
torch==2.11.0+cu128
```

如果没有 NVIDIA GPU，需要把 `requirements.txt` 里的 CUDA 版 `torch` 改成普通 CPU 版 `torch`。

WSL / Ubuntu RKNN 环境：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
python3 -m venv ~/.venvs/mindface-rknn
source ~/.venvs/mindface-rknn/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements-rknn.txt
python -m pip install -e .
python -m pip check
```

WSL 侧系统工具：

```bash
sudo apt update
sudo apt install -y git cmake make ninja-build pkg-config \
  device-tree-compiler \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

## 最短运行路径

### 0. 一键健康检查

在当前激活环境中运行：

```powershell
python scripts/99_health_check.py
```

也可以使用统一 CLI：

```powershell
python -m mindface health
```

输出：

```text
outputs/reports/health_check.json
```

`warn` 不一定是错误。例如在 WSL RKNN 环境里，麦克风或 MediaPipe 显示 `warn` 是正常的；在 Windows 训练环境里，`rknn-toolkit2` 显示 `warn` 也是正常的。

### 1. 基础 pipeline

在 Windows `mindface-lite` 中运行：

```powershell
conda activate mindface-lite
cd C:\Users\Administrator\Desktop\MindFace-Lite
python scripts/run_00_basic_pipeline.py
python scripts/15_verify_basic_outputs.py
```

基础 pipeline 会依次执行：

```text
生成测试音频 -> RMS 规则口型 demo -> Stage 1.5 better visual renderer
-> Stage 1.6 expressive static avatar -> 合成数据 -> MLP 训练
-> PyTorch 推理 -> ONNX 导出 -> ONNXRuntime 推理
-> PyTorch/ONNX/RKNN 可选一致性对比 -> 实时队列模拟
-> benchmark -> ONNX INT8 动态量化 -> 量化 benchmark
```

### 2. 手动运行基础步骤

```powershell
python scripts/00_generate_test_audio.py
python scripts/01_rule_mouth_demo.py --config configs/rule_demo.yaml
python scripts/01_5_better_visual_demo.py --config configs/better_visual_demo.yaml
python scripts/01_6_expressive_avatar_demo.py --config configs/expressive_avatar_demo.yaml
python scripts/02_generate_synthetic_dataset.py --config configs/synthetic_dataset.yaml
python scripts/03_train_model.py --config configs/train_mlp.yaml
python scripts/04_infer_pytorch.py --config configs/infer_pytorch.yaml
python scripts/05_export_onnx.py --config configs/export_onnx.yaml
python scripts/06_infer_onnx.py --config configs/infer_onnx.yaml
python scripts/07_realtime_rule_demo.py --config configs/realtime_rule.yaml
python scripts/08_benchmark.py --config configs/benchmark.yaml
```

对应的统一 CLI 常用入口：

```powershell
python -m mindface rule-demo
python -m mindface better-visual
python -m mindface expressive-avatar
python -m mindface train --config configs/train_mlp.yaml
python -m mindface export-onnx --config configs/export_onnx.yaml
python -m mindface compare-backends
```

### 3. GRID 音频预处理和训练

小样本 debug：

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/prepare_grid.yaml --max-samples 8 --output-dir data/processed/grid_mouth_debug
python scripts/03_train_model.py --config configs/train_grid_debug_mlp.yaml
```

全量 GRID 音频训练：

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/prepare_grid.yaml
python scripts/03_train_model.py --config configs/train_grid_mlp.yaml
```

这一路径使用 RMS 规则伪标签，适合验证真实音频数据扫描、特征提取和训练闭环。

每次训练都会自动生成实验追踪：

```text
outputs/experiments/<timestamp>_train_<model>/
├── config.yaml
├── history.csv
└── metrics.json
outputs/experiments/latest_train_run.txt
```

### 4. 量化和剪枝

```powershell
python scripts/10_quantize_onnx.py --config configs/quantize_onnx.yaml
python scripts/11_benchmark_quantized_onnx.py --config configs/benchmark_quantized_onnx.yaml
python scripts/12_prune_finetune.py --config configs/prune_finetune.yaml
python scripts/13_benchmark_pruned.py --config configs/benchmark_pruned.yaml
```

后端一致性对比：

```powershell
python scripts/17_compare_inference_backends.py --config configs/consistency_compare.yaml
```

报告输出：

```text
outputs/reports/backend_consistency_report.json
```

### 5. GRID 视频 landmark 标签

在 Windows `mindface-lite` 中运行：

```powershell
python scripts/14_extract_grid_video_landmarks.py --check-deps
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_debug
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --quality-only
```

当前默认配置：

```yaml
landmarks:
  delegate: cpu
```

说明：MediaPipe Python Tasks 暴露 `CPU/GPU` delegate，但当前 Windows pip wheel 实测 GPU graph 未启用，会报 `GPU processing is disabled in build flags`。因此 GRID landmark 批处理默认使用 CPU/XNNPACK。PyTorch 训练仍可使用 NVIDIA GPU。

质量报告默认输出：

```text
outputs/reports/grid_video_landmark_quality.json
```

将 GRID 视频 landmark 标签转换成真正的监督训练集：

```powershell
python scripts/16_prepare_grid_landmark_dataset.py --config configs/prepare_grid_landmark.yaml --max-samples 8
python scripts/03_train_model.py --config configs/train_grid_landmark_mlp.yaml
```

全量训练时去掉 `--max-samples 8`。该路径输出：

```text
data/processed/grid_landmark_mouth/manifest.csv
outputs/checkpoints/grid_landmark_mlp_mouth.pt
```

如果当前工作区没有 `data/raw/grid`，这一步会提示缺少 GRID 数据目录。

### 6. TTS 和麦克风

```powershell
python scripts/19_tts_generate_wav.py --config configs/tts_demo.yaml
python scripts/20_tts_mouth_demo.py --config configs/tts_demo.yaml
python scripts/21_real_tts_generate_wav.py --check-deps
python scripts/21_real_tts_generate_wav.py --config configs/real_tts.yaml
python scripts/22_real_tts_mouth_demo.py --config configs/real_tts.yaml
python scripts/23_mic_stream_rule_demo.py --check-deps
python scripts/23_mic_stream_rule_demo.py --list-devices
python scripts/23_mic_stream_rule_demo.py --config configs/mic_stream.yaml --duration-sec 10 --show
```

如果 `--list-devices` 能看到多个输入设备，可以在 `configs/mic_stream.yaml` 中设置：

```yaml
audio:
  device: 1
```

### 7. RKNN 和 Device Tree

在 WSL `mindface-rknn` 中运行：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python scripts/24_rknn_convert_and_infer.py --check-deps
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml --dry-run
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml
python scripts/25_device_tree_uboot_check.py --check-deps
python scripts/25_device_tree_uboot_check.py --config configs/device_tree_uboot.yaml
```

输出：

```text
outputs/models/mlp_mouth.rk3588.rknn
outputs/reports/rknn_deploy_report.json
outputs/embedded/rk3588-mouth-uart-overlay.dtbo
```

### 8. C++ demo

```powershell
cmake -S cpp -B build/cpp -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build/cpp
build\cpp\queue_demo.exe
build\cpp\udp_sender.exe outputs\logs\pytorch_mlp_params.csv 127.0.0.1 9000 25
build\cpp\serial_sender.exe outputs\logs\serial_output.txt outputs\logs\pytorch_mlp_params.csv 25
```

ONNXRuntime C++ demo 需要额外准备 ONNXRuntime C++ SDK：

```powershell
cmake -S cpp -B build/cpp-ort -DBUILD_ONNXRUNTIME_DEMO=ON -DONNXRUNTIME_DIR=C:\path\to\onnxruntime
```

## 输出文件

常见核心输出：

```text
outputs/audio/test_voice.wav
outputs/videos/rule_mouth_demo.mp4
outputs/videos/better_visual_mouth_demo.mp4
outputs/videos/expressive_avatar_demo.mp4
outputs/videos/expressive_avatar_preview.png
outputs/logs/rule_demo.csv
outputs/logs/better_visual_rule_demo.csv
outputs/logs/expressive_avatar_demo.csv
data/synthetic_mouth/manifest.csv
outputs/checkpoints/mlp_mouth.pt
outputs/models/mlp_mouth.onnx
outputs/models/mlp_mouth.int8.dynamic.onnx
outputs/checkpoints/grid_mlp_mouth.pt
outputs/checkpoints/grid_mlp_mouth.pruned.pt
outputs/experiments/latest_train_run.txt
outputs/reports/health_check.json
outputs/reports/benchmark_report.json
outputs/reports/quantized_onnx_benchmark.json
outputs/reports/pruned_benchmark.json
outputs/reports/grid_video_landmark_quality.json
outputs/reports/backend_consistency_report.json
outputs/models/mlp_mouth.rk3588.rknn
outputs/reports/rknn_deploy_report.json
outputs/embedded/rk3588-mouth-uart-overlay.dtbo
```

## 目录结构

```text
MindFace-Lite/
├── configs/                 # YAML 配置
├── assets/                  # 生成或准备的演示图片资产
├── data/                    # 原始数据和处理后数据
├── docs/                    # 分阶段文档、调试、面试说明
│   ├── requirements/        # 需求文档
│   └── features/            # 功能设计文档
├── outputs/                 # 音频、视频、日志、模型、报告输出
├── scripts/                 # 从项目根目录运行的脚本入口
├── src/mindface/            # Python 包源码
│   ├── audio/               # 音频读取和特征
│   ├── data/                # 数据集和 GRID 处理
│   ├── diagnostics/         # 健康检查
│   ├── deploy/              # ONNX / RKNN 工具
│   ├── experiments/         # 实验追踪
│   ├── models/              # MLP / LSTM / TCN / Transformer
│   ├── realtime/            # 实时队列和麦克风
│   ├── tts/                 # 伪 TTS 和真实 TTS
│   ├── utils/               # 配置、日志、CSV
│   └── visual/              # 嘴巴动画绘制
├── cpp/                     # C++ 实时控制 demo
├── tests/                   # pytest 自动测试
├── tools/                   # 嵌入式和交叉编译辅助
├── PROJECT_DEMO.md          # 面试/演示速查稿
├── pyproject.toml           # 包安装、CLI、pytest 配置
├── requirements.txt
├── requirements-optional.txt
├── requirements-dev.txt
├── requirements-rknn.txt
└── RUN_ORDER.md
```

## 岗位能力映射

| 岗位要求 | 项目对应内容 |
| --- | --- |
| AI/ML 基础 | MLP、LSTM、TCN、Transformer；训练、推理、loss、优化器 |
| PyTorch 训练闭环 | Dataset、DataLoader、train/val、checkpoint、inference |
| 部署能力 | ONNX 导出、ONNXRuntime、benchmark、RKNN 转换 |
| 模型优化 | ONNX INT8 dynamic quantization、剪枝、fine-tune、对比报告 |
| Python 工程 | YAML 配置、logging、异常提示、脚本入口、性能统计 |
| 实验管理 | 自动保存 config、history、metrics、健康检查和部署报告 |
| 测试能力 | pytest 覆盖音频特征、数据集、GRID landmark 预处理、视觉渲染、后端误差计算 |
| 实时工程 | 麦克风 streaming、队列、FPS、latency、UDP 输出 |
| C++ 基础 | CMake、多线程队列、UDP/串口输出、ONNXRuntime C++ demo 入口 |
| 边缘部署思维 | RK3588、RKNN、NPU、Device Tree、U-Boot、交叉编译 |
| 文档能力 | 安装、运行顺序、调试、部署、性能、面试说明 |

## 学习路线

推荐按下面顺序学习：

1. `docs/00_rule_demo.md`：理解 RMS 规则系统和嘴巴动画。
2. `docs/01_training.md`：理解 PyTorch 训练闭环。
3. `docs/02_deployment.md`：理解 ONNX / ONNXRuntime 部署。
4. `docs/09_grid_preprocessing.md`：理解真实数据预处理和 manifest。
5. `docs/10_quantization_pruning.md`：理解量化、剪枝和 benchmark。
6. `docs/03_realtime_engineering.md`：理解实时队列、延迟和 FPS。
7. `docs/04_cpp_control.md`：理解 C++ 控制侧 demo。
8. `docs/11_landmark_tts_mic_rknn.md`：理解 landmark、TTS、麦克风、RKNN、Device Tree。
9. `docs/07_interview_notes.md`：整理成面试表达。

完整命令顺序见：

- `RUN_ORDER.md`

常见错误见：

- `docs/06_debugging.md`

## 文档索引

- `docs/00_rule_demo.md`：RMS 规则 demo、公式、常见错误、面试解释。
- `docs/requirements/001-mindface-lite.md`：项目总需求和验收范围。
- `docs/features/environment-matrix.md`：Windows/WSL/硬件环境职责矩阵。
- `docs/features/unified-cli.md`：统一 CLI 入口。
- `docs/features/backend-consistency.md`：PyTorch、ONNXRuntime、RKNN 后端一致性对比。
- `docs/features/better-visual-renderer.md`：Stage 1.5 更精致的 OpenCV 口型渲染器。
- `docs/features/expressive-static-avatar.md`：Stage 1.6 静态人脸图 + OpenCV 嘴部 ROI 形变。
- `docs/features/health-check.md`：一键健康检查设计。
- `docs/features/experiment-tracking.md`：训练实验追踪设计。
- `docs/features/grid-landmark-quality.md`：GRID landmark 标签质量报告设计。
- `docs/features/rknn-deployment-report.md`：RKNN 部署报告设计。
- `docs/01_training.md`：合成数据集和 PyTorch 训练闭环。
- `docs/02_deployment.md`：PyTorch 推理、ONNX 导出、ONNXRuntime、benchmark。
- `docs/03_realtime_engineering.md`：队列、延迟、FPS、实时工程取舍。
- `docs/04_cpp_control.md`：CMake、C++ 队列、UDP / 串口执行器控制。
- `docs/05_rk3588_rknn_plan.md`：RK3588、RKNN、INT8、硬件约束。
- `docs/06_debugging.md`：常见错误和修复方法。
- `docs/07_interview_notes.md`：面向面试的项目讲解。
- `docs/08_performance_report.md`：性能报告摘要。
- `docs/09_grid_preprocessing.md`：GRID 预处理和 manifest 生成说明。
- `docs/10_quantization_pruning.md`：ONNX INT8 动态量化、剪枝、fine-tune、benchmark。
- `docs/11_landmark_tts_mic_rknn.md`：GRID 视频 landmark、真实 TTS、麦克风、RKNN、Device Tree / U-Boot 入口。
- `docs/grid-landmark-training-report.md`：真实 GRID landmark 训练报告模板，数据恢复后填写 loss、样本数、检测率和后端一致性。
- `docs/media/README.md`：README 和文档用的小型演示图片索引。
- `docs/file-disposition-review.md`：项目文件归档与清理记录。
- `PROJECT_DEMO.md`：项目演示顺序、输出查看和面试表达速查。

## License

本项目使用 MIT License，见 `LICENSE`。

## 当前边界

MindFace-Lite 当前重点是工程链路完整性，而不是生产级口型效果。它已经覆盖“从规则系统到训练、部署、优化、实时控制、边缘部署思考”的主线，但仍有这些边界：

- GRID 音频训练默认仍是 RMS 伪标签，不是完全真实的嘴部运动监督。
- GRID 视频 landmark 已有提取入口，但真正训练到视频口型标签还需要完成音视频对齐和训练配置。
- MediaPipe Windows pip 包当前不支持 GPU graph，landmark 批处理走 CPU/XNNPACK。
- RKNN 转换已在 WSL 中验证，板端实机推理仍需要真实 RK3588 设备、RKNN Runtime 和硬件接口联调。
- Device Tree / U-Boot 脚本只生成和展示操作命令，不会自动修改真实板子的 BSP。
