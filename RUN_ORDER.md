# 运行顺序

所有命令都建议在项目根目录运行：

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
```

## 0. 安装环境

已经安装过 editable package 时，推荐使用方向键菜单：

```powershell
conda activate mindface-lite
mindface ui
```

首次安装还没有 `mindface` 命令时：

```powershell
$env:PYTHONPATH = "src"
python -m mindface ui
```

进入“环境与安装”，先运行完整检查。命令行等价入口：

```powershell
mindface env status --distro Ubuntu
mindface env check --distro Ubuntu
mindface env install-windows --source official --dry-run
mindface env install-wsl --distro Ubuntu --source official --dry-run
```

确认计划后去掉 `--dry-run`。安装器只创建缺失环境或修复已有环境，不自动删除；WSL 的 `sudo apt` 阶段会要求输入 Ubuntu 密码。

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

安装后以 `mindface ui` 作为日常入口。`scripts/*.py` 仅保留为旧教程和旧命令的兼容跳转，不再包含业务逻辑。

当前 `requirements.txt` 默认安装 CUDA 版 PyTorch，适合 NVIDIA GPU 训练。如果没有 NVIDIA GPU，需要把其中的 `torch==2.11.0+cu128` 改成普通 `torch`。

可选功能依赖：

```powershell
python -m pip install -r requirements-optional.txt
python -m pip install -r requirements-dev.txt
```

依赖文件不会无脑追最新版本。`requirements.txt` / `requirements-optional.txt` 会保持 `numpy<2.0`，并固定 OpenCV `4.11.0.86`，避免 `opencv-contrib-python 5.x` 把环境升级到 NumPy 2.x。MediaPipe、TTS、麦克风建议在 Windows `mindface-lite` 环境运行；RKNN 仍然使用单独的 `requirements-rknn.txt`。

WSL/Ubuntu 的 RKNN 转换环境不要使用 `requirements.txt`，应使用：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python -m pip install -r requirements-rknn.txt
python -m pip install -e .
python -m pip check
```

不要在 `mindface-rknn` 里安装 `requirements-optional.txt`。MediaPipe / `opencv-contrib-python` 用于 Windows 训练环境的 GRID landmark 提取；RKNN 环境要保持 `numpy==1.26.4`，否则 `rknn-toolkit2` 可能报依赖或 ONNX 转换错误。

`requirements-rknn.txt` 也固定了 `setuptools==80.9.0`，因为 `rknn-toolkit2==2.3.2` 仍会导入旧的 `pkg_resources`。

## 1. 先跑一键健康检查

```powershell
mindface ui
mindface health
mindface config validate --all
```

输出：

```text
outputs/reports/health_check.json
```

健康检查会列出 Python 包、CUDA、关键数据、关键输出、外部工具状态。`warn` 表示当前环境可能不负责该功能，例如 Windows 训练环境缺少 RKNN 或 WSL RKNN 环境缺少麦克风都可以接受。

## 2. 推荐先跑基础 Pipeline

```powershell
mindface pipeline basic
mindface verify
```

当前基础 pipeline 会跑规则 demo、Stage 1.5 better visual renderer、Stage 1.6 expressive static avatar、基础训练、推理、ONNX 导出、ONNXRuntime、PyTorch/ONNX/RKNN 可选一致性对比、实时队列、benchmark，并默认加入 ONNX INT8 动态量化和量化 benchmark。

可选参数：

```powershell
mindface pipeline basic --skip-quantization
mindface pipeline basic --include-grid-compression
mindface pipeline basic --check-optional-deps
mindface pipeline basic --check-optional-deps-only
mindface pipeline basic --dry-run
mindface pipeline basic --from-step train --to-step benchmark
mindface pipeline basic --from-step train --to-step benchmark --force
```

注意：这个基础 pipeline 不会预处理或训练全量 GRID 数据。`--include-grid-compression` 只会使用已经训练好的 GRID checkpoint 做导出、量化、剪枝和 benchmark。

如果想用统一 CLI 手动跑关键阶段：

```powershell
mindface demo rule
mindface demo better-visual
mindface demo expressive-avatar
mindface train --config configs/training/train-mlp.yaml
mindface export onnx
mindface benchmark backends
```

## 3. 手动运行基础 Pipeline

```powershell
mindface demo generate-audio
mindface demo rule --config configs/demos/rule-demo.yaml
mindface demo better-visual --config configs/demos/better-visual-demo.yaml
mindface demo expressive-avatar --config configs/demos/expressive-avatar-demo.yaml
mindface data synthetic --config configs/datasets/synthetic-dataset.yaml
mindface train --config configs/training/train-mlp.yaml
mindface infer pytorch --config configs/inference/infer-pytorch.yaml
mindface export onnx --config configs/deployment/export-onnx.yaml
mindface infer onnx --config configs/inference/infer-onnx.yaml
mindface realtime queue --config configs/realtime/realtime-rule.yaml
mindface benchmark runtime --config configs/benchmarks/benchmark.yaml
mindface verify
```

训练脚本会自动生成实验追踪目录：

```text
outputs/experiments/<timestamp>_train_<model>/
├── config.yaml
├── history.csv
└── metrics.json
```

恢复训练：编辑训练 YAML，把 `train.resume_from` 设置为对应的 `output.last_checkpoint_path`，并将 `train.epochs` 改成新的总 epoch 数。每个训练配置默认写入 best `.pt` 和 resumable `.last.pt`。

## 4. 可选：训练其他模型

```powershell
mindface train --config configs/training/train-lstm.yaml
mindface train --config configs/training/train-tcn.yaml
mindface train --config configs/training/train-transformer.yaml
```

## 5. GRID 小样本 Debug 预处理和训练

建议在全量 GRID 之前先跑这个：

```powershell
mindface data prepare-grid --config configs/datasets/prepare-grid.yaml --max-samples 8 --output-dir data/processed/grid_mouth_debug
mindface train --config configs/training/train-grid-debug-mlp.yaml
```

## 6. 全量 GRID 音频预处理和训练

这一步会扫描并处理 `data/raw/grid` 下的 GRID 原始音频，耗时和磁盘占用都会更高。

```powershell
mindface data prepare-grid --config configs/datasets/prepare-grid.yaml
mindface train --config configs/training/train-grid-mlp.yaml
```

当前标签是基于 RMS 规则生成的伪嘴型标签。真正的视频嘴型标签需要后续实现 landmark / blendshape 提取。

## 6.5 GRID Landmark 真实标签预处理和训练

这一步需要真实数据存在：

```text
data/raw/grid/audio
data/raw/grid/video
```

先提取视频 landmark：

```powershell
mindface data extract-landmarks --check-deps
mindface data extract-landmarks --config configs/datasets/grid-video-landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_debug
mindface data extract-landmarks --config configs/datasets/grid-video-landmarks.yaml
```

再把 `data/processed/grid_video_landmarks` 中的 target 和 `data/raw/grid/audio` 中的 WAV 对齐成训练集：

```powershell
mindface data align-landmarks --config configs/datasets/prepare-grid-landmark.yaml --max-samples 8
mindface data align-landmarks --config configs/datasets/prepare-grid-landmark.yaml
```

训练真正以 landmark 参数为监督的 MLP：

```powershell
mindface train --config configs/training/train-grid-landmark-mlp.yaml
```

输出：

```text
data/processed/grid_landmark_mouth/manifest.csv
outputs/checkpoints/grid_landmark_mlp_mouth.pt
```

如果当前项目目录没有 `data/raw/grid`，这一步不能真实训练，会提示缺少 GRID 数据目录。

## 7. 量化与剪枝

ONNX INT8 动态量化和对比 benchmark：

```powershell
mindface optimize quantize --config configs/optimization/quantize-onnx.yaml
mindface optimize benchmark-quantized --config configs/benchmarks/benchmark-quantized-onnx.yaml
```

GRID MLP 量化：

```powershell
mindface export onnx --config configs/deployment/export-grid-onnx.yaml
mindface optimize quantize --config configs/optimization/quantize-grid-onnx.yaml
mindface optimize benchmark-quantized --config configs/benchmarks/benchmark-grid-quantized-onnx.yaml
```

PyTorch 剪枝、fine-tune 和对比 benchmark：

```powershell
mindface optimize prune --config configs/optimization/prune-finetune.yaml
mindface optimize benchmark-pruned --config configs/benchmarks/benchmark-pruned.yaml
```

PyTorch、ONNXRuntime、RKNN 可选一致性对比：

```powershell
mindface benchmark backends --config configs/benchmarks/backend-consistency.yaml
```

报告：

```text
outputs/reports/backend_consistency_report.json
```

## 8. GRID 视频 Landmark 标签

先检查可选依赖：

```powershell
mindface data extract-landmarks --check-deps
```

小样本调试：

```powershell
mindface data extract-landmarks --config configs/datasets/grid-video-landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_debug
```

全量提取：

```powershell
mindface data extract-landmarks --config configs/datasets/grid-video-landmarks.yaml
```

生成或刷新已有 landmark 的质量报告：

```powershell
mindface data extract-landmarks --config configs/datasets/grid-video-landmarks.yaml --quality-only
```

报告输出：

```text
outputs/reports/grid_video_landmark_quality.json
```

## 9. TTS-like Demo

```powershell
mindface tts pseudo-generate --config configs/realtime/tts-demo.yaml
mindface tts pseudo-demo --config configs/realtime/tts-demo.yaml
```

## 10. 真实 TTS 与麦克风 Streaming

真实 TTS：

```powershell
mindface tts generate --check-deps
mindface tts generate --config configs/realtime/real-tts.yaml
mindface tts demo --config configs/realtime/real-tts.yaml
```

麦克风 streaming：

```powershell
mindface realtime microphone --check-deps
mindface realtime microphone --config configs/realtime/mic-stream.yaml --duration-sec 10 --show
```

## 11. RKNN 与 Device Tree / U-Boot

Windows 环境只做依赖检查或 dry-run：

```powershell
mindface deploy rknn --check-deps
mindface deploy rknn --config configs/deployment/rknn-deploy.yaml --dry-run
```

WSL/Ubuntu `mindface-rknn` 环境做真实 RKNN 转换：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python -m pip install -r requirements-rknn.txt
python -m pip check
mindface deploy rknn --check-deps
mindface deploy rknn --config configs/deployment/rknn-deploy.yaml
```

RKNN dry-run 和真实转换都会写部署报告：

```text
outputs/reports/rknn_deploy_report.json
```

Device Tree / U-Boot：

```powershell
mindface deploy device-tree --check-deps
mindface deploy device-tree --config configs/deployment/device-tree-uboot.yaml
```

## 12. C++ Demo

```powershell
mindface cpp configure
mindface cpp build
mindface cpp test
mindface cpp run queue-demo
mindface cpp run udp-sender -- outputs\logs\pytorch_mlp_params.csv 127.0.0.1 9000 25
mindface cpp run serial-sender -- outputs\logs\serial_output.txt outputs\logs\pytorch_mlp_params.csv 25
```

## 13. 自动测试

```powershell
python -m pytest
```

测试覆盖音频 RMS、manifest 数据集、GRID landmark 预处理小样本、Stage 1.6 渲染器和后端误差计算。
