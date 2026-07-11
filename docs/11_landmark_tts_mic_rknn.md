# 11. Landmark、真实 TTS、麦克风、RKNN 与板端实操入口

本阶段补齐之前“待做”的工程入口。部分功能依赖外部库或硬件，当前项目提供可运行脚本、依赖检查、配置文件和清晰错误提示。

## 可选依赖

```powershell
python -m pip install -r requirements-optional.txt
```

这些可选依赖建议安装在 Windows conda 环境 `mindface-lite` 中。不要安装到 WSL/Ubuntu 的 `mindface-rknn`，因为 RKNN-Toolkit2 需要稳定的 `numpy<=1.26.4`，而 `mediapipe` 在 Linux 上可能拉取 `opencv-contrib-python` 并升级 `numpy`。

如果需要在 WSL 中单独调试 MediaPipe、真实 TTS 或麦克风，请新建另一个 venv，不要复用 `mindface-rknn`。常用系统依赖：

```bash
sudo apt install -y libgles2 libegl1 portaudio19-dev libportaudio2 ffmpeg espeak-ng
```

说明：

- `mediapipe`：GRID 视频 face landmark 提取。
- `pyttsx3`：离线系统 TTS。
- `edge-tts`：在线神经 TTS，输出转 WAV 需要 FFmpeg。
- `sounddevice`：麦克风 streaming 输入。
- `rknn-toolkit2`：Rockchip RKNN 转换，需要在 RKNN 开发环境中安装。
- `dtc`：Device Tree Compiler，通常在 Linux/BSP 环境中使用。

## GRID 视频 Landmark 标签

当前脚本使用 MediaPipe Face Landmarker Tasks API。新版 `mediapipe 0.10.35` 不再暴露旧的 `mp.solutions.face_mesh` 入口，所以代码会使用：

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
```

首次运行时会自动下载官方 `face_landmarker.task` 模型到：

```text
models/mediapipe/face_landmarker.task
```

检查依赖：

```powershell
python scripts/14_extract_grid_video_landmarks.py --check-deps
```

推荐在 Windows `mindface-lite` 环境运行 GRID landmark 提取：

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
conda activate mindface-lite
python -m pip install -r requirements-optional.txt
python scripts/14_extract_grid_video_landmarks.py --check-deps
```

不要在 WSL `mindface-rknn` 环境安装 `mediapipe`。如果必须在 WSL 中跑 landmark，请新建单独环境，例如 `~/.venvs/mindface-media`，不要复用 RKNN 环境。

MediaPipe Python Tasks 支持 `BaseOptions.Delegate.CPU/GPU`，项目配置中可以设置：

```yaml
landmarks:
  delegate: cpu
```

也可以临时测试 GPU：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_gpu_debug --delegate gpu
```

注意：GPU delegate 在 Python 中有平台限制。Google 的 `BaseOptions` 文档说明 GPU support 目前限制在 Ubuntu 平台；当前 Windows `mindface-lite` 实测会报 `GPU processing is disabled in build flags`。如果 GPU delegate 报错，把 `delegate` 改回 `cpu`。当前批量 GRID 提取默认使用 CPU/XNNPACK，这是稳定路线。

WSL 中如果看到 `libGLESv2.so.2: cannot open shared object file`，说明 MediaPipe 已安装，但系统 OpenGL ES/EGL 运行库缺失：

```bash
sudo apt install -y libgles2 libegl1
```

小样本调试：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_debug
```

全量提取：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml
```

输出：

```text
data/processed/grid_video_landmarks/manifest.csv
data/processed/grid_video_landmarks/landmarks/*.csv
data/processed/grid_video_landmarks/targets_*.npy
models/mediapipe/face_landmarker.task
outputs/reports/grid_video_landmark_quality.json
```

如果 landmark 已经提取过，可以单独刷新质量报告，不需要重复跑 MediaPipe：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --quality-only
```

质量报告会统计整体检测率、每个样本检测率、`mouth_open/mouth_width/mouth_round` 分布，以及低检测样本列表。它用于判断 landmark 标签是否可靠，但不代表音频和嘴型已经完成严格监督对齐。

标签含义：

- `mouth_open`：上下唇 landmark 距离，按脸宽归一化。
- `mouth_width`：嘴角距离，按脸宽归一化。
- `mouth_round`：嘴部纵横比，表示圆唇程度。

## 使用 Landmark 标签训练

完成 `data/processed/grid_video_landmarks` 后，可以把视频 landmark 标签和原始 WAV 音频对齐成训练集：

```powershell
python scripts/16_prepare_grid_landmark_dataset.py --config configs/prepare_grid_landmark.yaml --max-samples 8
```

确认小样本输出没问题后，跑全量：

```powershell
python scripts/16_prepare_grid_landmark_dataset.py --config configs/prepare_grid_landmark.yaml
python scripts/03_train_model.py --config configs/train_grid_landmark_mlp.yaml
```

输出：

```text
data/processed/grid_landmark_mouth/manifest.csv
outputs/checkpoints/grid_landmark_mlp_mouth.pt
```

这条路径和 `scripts/09_prepare_grid_dataset.py` 不同：`09` 使用 RMS 规则伪标签，`16` 使用视频 landmark 生成的 `mouth_open/mouth_width/mouth_round` 作为监督标签。

## 真实 TTS

只生成 WAV：

```powershell
python scripts/21_real_tts_generate_wav.py --config configs/real_tts.yaml
```

生成真实 TTS 音频并驱动规则嘴巴动画：

```powershell
python scripts/22_real_tts_mouth_demo.py --config configs/real_tts.yaml
```

说明：Linux/WSL 的 `pyttsx3` 通常走 espeak 后端，可能在 Python 退出时出现 `_enter_buffered_busy` 或 `EspeakDriver._onSynth` 回调错误。项目配置 `pyttsx3.isolate_subprocess: auto` 会在 Linux 下把 pyttsx3 放到子进程里生成 WAV，主流程只检查输出文件是否有效，再继续生成 CSV 和 MP4。推荐仍然在 Windows `mindface-lite` 环境运行 TTS，不要把 `mindface-rknn` 当成通用可选功能环境。

输出：

```text
outputs/audio/real_tts.wav
outputs/videos/real_tts_rule_mouth_demo.mp4
outputs/logs/real_tts_rule_demo.csv
```

## 麦克风 Streaming 输入

检查依赖：

```powershell
python scripts/23_mic_stream_rule_demo.py --check-deps
```

列出当前环境可见的音频设备：

```powershell
python scripts/23_mic_stream_rule_demo.py --list-devices
```

WSL 中如果看到 `PortAudio library not found`，说明 Python 包已安装但系统 PortAudio 缺失：

```bash
sudo apt install -y portaudio19-dev libportaudio2
```

如果看到 `Error querying device -1` 或 `default.device=[-1, -1]`，说明 PortAudio 没有看到默认输入设备。优先在 Windows `mindface-lite` 环境运行麦克风 streaming；如果 `--list-devices` 能列出输入设备，再把对应编号或名称写入 `configs/mic_stream.yaml` 的 `audio.device`。

运行 10 秒：

```powershell
python scripts/23_mic_stream_rule_demo.py --config configs/mic_stream.yaml --duration-sec 10 --show
```

输出：

```text
outputs/logs/mic_stream_rule_demo.csv
```

脚本使用声卡 callback 采集音频块，主线程计算 RMS、平滑 `mouth_open`、写 CSV，并支持可选 UDP 输出。

## RKNN 转换与推理入口

检查依赖：

```powershell
python scripts/24_rknn_convert_and_infer.py --check-deps
```

在 Windows 上只检查配置和计划步骤：

```powershell
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml --dry-run
```

Windows 环境只建议做 dry-run。真正转换 ONNX 到 RKNN 请使用 WSL/Ubuntu `mindface-rknn` 环境：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python -m pip install -r requirements-rknn.txt
python -m pip check
python scripts/24_rknn_convert_and_infer.py --check-deps
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml
```

启用 RKNN build-time 量化：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml --quantize
```

输出：

```text
outputs/models/mlp_mouth.rk3588.rknn
outputs/reports/rknn_deploy_report.json
```

RKNN 转换注意事项：

- RKNN-Toolkit2 2.3.2 在 Python 3.12 + `onnx 1.22` 下可能报 `module 'onnx' has no attribute 'mapping'`。项目已在 `src/mindface/deploy/rknn_tools.py` 中加入兼容 shim，不需要手动修改 site-packages。
- RKNN 不接受 ONNX 动态输入 `['frames', 70]`。项目会在 `rknn.load_onnx()` 中固定输入 shape，默认来自 `configs/rknn_deploy.yaml`：

```yaml
input:
  frames: 16
  feature_dim: 70
```

注意：`rknn-toolkit2` 通常安装在 Ubuntu x86_64 的 RKNN 开发环境或 Rockchip SDK Docker 中，不是在 Windows 训练环境中直接安装。Windows 环境可以用 `--dry-run` 验证配置；真正转换请切到 Ubuntu/RKNN 环境。RK3588 板端部署使用 RKNN-Toolkit-Lite2 或 RKNN Runtime。

注意：不要在 `mindface-rknn` 环境里安装 `requirements-optional.txt`。`mediapipe` / `opencv-contrib-python` 用于 Windows 训练环境的 GRID landmark 提取，可能把 `numpy` 升到 2.x；`rknn-toolkit2==2.3.2` 需要 `numpy<=1.26.4`。如果误装，可以执行：

```bash
python -m pip uninstall -y mediapipe opencv-contrib-python
python -m pip install --force-reinstall -r requirements-rknn.txt
python -m pip check
```

`requirements-rknn.txt` 会固定 `setuptools==80.9.0`，因为 `rknn-toolkit2==2.3.2` 仍依赖旧的 `pkg_resources` 模块。若看到 `ModuleNotFoundError: No module named 'pkg_resources'`，通常就是 `setuptools` 版本过新。

注意：RKNN 量化最好使用真实音频特征校准集。当前脚本可以自动生成 dummy 校准样本用于流程验证，但正式部署应使用 GRID/TTS/麦克风采样得到的代表性特征。

RKNN 部署报告会记录 Python、NumPy、ONNX、ONNXRuntime、RKNN-Toolkit2 版本，ONNX/RKNN 文件大小，输入 shape，量化数据集条目数，以及下一步板端验证建议。即使使用 `--dry-run`，也会生成该报告，便于先检查配置和环境边界。

## Device Tree / U-Boot 实操辅助

检查 `dtc`：

```powershell
python scripts/25_device_tree_uboot_check.py --check-deps
```

编译 overlay：

```powershell
python scripts/25_device_tree_uboot_check.py --config configs/device_tree_uboot.yaml
```

输入：

```text
tools/embedded/rk3588-mouth-uart-overlay.dts
```

输出：

```text
outputs/embedded/rk3588-mouth-uart-overlay.dtbo
```

脚本会打印 U-Boot 中加载 dtb、加载 overlay、`fdt apply`、启动内核的示例命令，以及 Linux 侧检查 UART 设备节点的命令。

## 面试解释

我把真实工程中的外部依赖和硬件依赖拆成可验证入口：landmark 标签用于从音视频数据获得监督信号，TTS 和麦克风用于真实输入，RKNN 脚本用于 SoC 部署链路，Device Tree/U-Boot 脚本用于和 BSP/硬件团队对接接口配置。即使没有板子，也能说明每一步的输入、输出、依赖和失败点。
