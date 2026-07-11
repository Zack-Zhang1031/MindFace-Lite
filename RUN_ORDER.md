# 运行顺序

所有命令都建议在项目根目录运行：

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
```

## 0. 安装环境

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

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
python scripts/99_health_check.py
```

统一 CLI 等价命令：

```powershell
python -m mindface health
```

输出：

```text
outputs/reports/health_check.json
```

健康检查会列出 Python 包、CUDA、关键数据、关键输出、外部工具状态。`warn` 表示当前环境可能不负责该功能，例如 Windows 训练环境缺少 RKNN 或 WSL RKNN 环境缺少麦克风都可以接受。

## 2. 推荐先跑基础 Pipeline

```powershell
python scripts/run_00_basic_pipeline.py
python scripts/15_verify_basic_outputs.py
```

当前基础 pipeline 会跑规则 demo、Stage 1.5 better visual renderer、Stage 1.6 expressive static avatar、基础训练、推理、ONNX 导出、ONNXRuntime、PyTorch/ONNX/RKNN 可选一致性对比、实时队列、benchmark，并默认加入 ONNX INT8 动态量化和量化 benchmark。

可选参数：

```powershell
python scripts/run_00_basic_pipeline.py --skip-quantization
python scripts/run_00_basic_pipeline.py --include-grid-compression
python scripts/run_00_basic_pipeline.py --check-optional-deps
python scripts/run_00_basic_pipeline.py --check-optional-deps-only
```

注意：这个基础 pipeline 不会预处理或训练全量 GRID 数据。`--include-grid-compression` 只会使用已经训练好的 GRID checkpoint 做导出、量化、剪枝和 benchmark。

如果想用统一 CLI 手动跑关键阶段：

```powershell
python -m mindface rule-demo
python -m mindface better-visual
python -m mindface expressive-avatar
python -m mindface train --config configs/train_mlp.yaml
python -m mindface export-onnx --config configs/export_onnx.yaml
python -m mindface compare-backends
```

## 3. 手动运行基础 Pipeline

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
python scripts/15_verify_basic_outputs.py
```

训练脚本会自动生成实验追踪目录：

```text
outputs/experiments/<timestamp>_train_<model>/
├── config.yaml
├── history.csv
└── metrics.json
```

## 4. 可选：训练其他模型

```powershell
python scripts/03_train_model.py --config configs/train_lstm.yaml
python scripts/03_train_model.py --config configs/train_tcn.yaml
python scripts/03_train_model.py --config configs/train_transformer.yaml
```

## 5. GRID 小样本 Debug 预处理和训练

建议在全量 GRID 之前先跑这个：

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/prepare_grid.yaml --max-samples 8 --output-dir data/processed/grid_mouth_debug
python scripts/03_train_model.py --config configs/train_grid_debug_mlp.yaml
```

## 6. 全量 GRID 音频预处理和训练

这一步会扫描并处理 `data/raw/grid` 下的 GRID 原始音频，耗时和磁盘占用都会更高。

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/prepare_grid.yaml
python scripts/03_train_model.py --config configs/train_grid_mlp.yaml
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
python scripts/14_extract_grid_video_landmarks.py --check-deps
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_debug
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml
```

再把 `data/processed/grid_video_landmarks` 中的 target 和 `data/raw/grid/audio` 中的 WAV 对齐成训练集：

```powershell
python scripts/16_prepare_grid_landmark_dataset.py --config configs/prepare_grid_landmark.yaml --max-samples 8
python scripts/16_prepare_grid_landmark_dataset.py --config configs/prepare_grid_landmark.yaml
```

训练真正以 landmark 参数为监督的 MLP：

```powershell
python scripts/03_train_model.py --config configs/train_grid_landmark_mlp.yaml
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
python scripts/10_quantize_onnx.py --config configs/quantize_onnx.yaml
python scripts/11_benchmark_quantized_onnx.py --config configs/benchmark_quantized_onnx.yaml
```

GRID MLP 量化：

```powershell
python scripts/05_export_onnx.py --config configs/export_grid_onnx.yaml
python scripts/10_quantize_onnx.py --config configs/quantize_grid_onnx.yaml
python scripts/11_benchmark_quantized_onnx.py --config configs/benchmark_grid_quantized_onnx.yaml
```

PyTorch 剪枝、fine-tune 和对比 benchmark：

```powershell
python scripts/12_prune_finetune.py --config configs/prune_finetune.yaml
python scripts/13_benchmark_pruned.py --config configs/benchmark_pruned.yaml
```

PyTorch、ONNXRuntime、RKNN 可选一致性对比：

```powershell
python scripts/17_compare_inference_backends.py --config configs/consistency_compare.yaml
```

报告：

```text
outputs/reports/backend_consistency_report.json
```

## 8. GRID 视频 Landmark 标签

先检查可选依赖：

```powershell
python scripts/14_extract_grid_video_landmarks.py --check-deps
```

小样本调试：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_debug
```

全量提取：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml
```

生成或刷新已有 landmark 的质量报告：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --quality-only
```

报告输出：

```text
outputs/reports/grid_video_landmark_quality.json
```

## 9. TTS-like Demo

```powershell
python scripts/19_tts_generate_wav.py --config configs/tts_demo.yaml
python scripts/20_tts_mouth_demo.py --config configs/tts_demo.yaml
```

## 10. 真实 TTS 与麦克风 Streaming

真实 TTS：

```powershell
python scripts/21_real_tts_generate_wav.py --check-deps
python scripts/21_real_tts_generate_wav.py --config configs/real_tts.yaml
python scripts/22_real_tts_mouth_demo.py --config configs/real_tts.yaml
```

麦克风 streaming：

```powershell
python scripts/23_mic_stream_rule_demo.py --check-deps
python scripts/23_mic_stream_rule_demo.py --config configs/mic_stream.yaml --duration-sec 10 --show
```

## 11. RKNN 与 Device Tree / U-Boot

Windows 环境只做依赖检查或 dry-run：

```powershell
python scripts/24_rknn_convert_and_infer.py --check-deps
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml --dry-run
```

WSL/Ubuntu `mindface-rknn` 环境做真实 RKNN 转换：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python -m pip install -r requirements-rknn.txt
python -m pip check
python scripts/24_rknn_convert_and_infer.py --check-deps
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml
```

RKNN dry-run 和真实转换都会写部署报告：

```text
outputs/reports/rknn_deploy_report.json
```

Device Tree / U-Boot：

```powershell
python scripts/25_device_tree_uboot_check.py --check-deps
python scripts/25_device_tree_uboot_check.py --config configs/device_tree_uboot.yaml
```

## 12. C++ Demo

```powershell
cmake -S cpp -B build/cpp -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
cmake --build build/cpp
build\cpp\queue_demo.exe
build\cpp\udp_sender.exe outputs\logs\pytorch_mlp_params.csv 127.0.0.1 9000 25
build\cpp\serial_sender.exe outputs\logs\serial_output.txt outputs\logs\pytorch_mlp_params.csv 25
```

## 13. 自动测试

```powershell
python -m pytest
```

测试覆盖音频 RMS、manifest 数据集、GRID landmark 预处理小样本、Stage 1.6 渲染器和后端误差计算。
