# Debugging Guide

## Python Import Errors

Symptom:

```text
ModuleNotFoundError: No module named 'mindface'
```

Fix:

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
python -m pip install -e .
```

The provided scripts also inject `src/` into `sys.path` when run from the project root.

## Missing Audio

Symptom:

```text
Missing input WAV
```

Fix:

```powershell
python scripts/00_generate_test_audio.py
```

## Missing Dataset

Symptom:

```text
Dataset manifest not found
```

Fix:

```powershell
python scripts/02_generate_synthetic_dataset.py --config configs/synthetic_dataset.yaml
```

## Missing GRID Raw Data

Symptom:

```text
GRID audio directory not found
GRID video directory not found
```

Fix:

Copy or restore the GRID corpus under:

```text
data/raw/grid/audio
data/raw/grid/video
data/raw/grid/alignments
```

Then rerun the intended preprocessing step:

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/prepare_grid.yaml
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml
python scripts/16_prepare_grid_landmark_dataset.py --config configs/prepare_grid_landmark.yaml
```

`09_prepare_grid_dataset.py` builds RMS pseudo labels from GRID audio. `16_prepare_grid_landmark_dataset.py` requires landmark outputs from `14_extract_grid_video_landmarks.py`.

## Missing Checkpoint

Symptom:

```text
Checkpoint not found
```

Fix:

```powershell
python scripts/03_train_model.py --config configs/train_mlp.yaml
```

## ONNXRuntime Not Installed

Symptom:

```text
onnxruntime is not installed
```

Fix:

```powershell
python -m pip install onnxruntime
```

## RKNN Environment Contaminated

Symptom:

```text
rknn-toolkit2 2.3.2 requires numpy<=1.26.4, but you have numpy 2.x
```

Cause:

`mediapipe` or `opencv-contrib-python` was installed into the WSL RKNN environment.

Fix:

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python -m pip uninstall -y mediapipe opencv-contrib-python
python -m pip install --force-reinstall -r requirements-rknn.txt
python -m pip check
```

Use `requirements-optional.txt` only in the Windows training environment, not in `mindface-rknn`.

Do not solve this by upgrading everything to the latest versions. As of the current project setup, PyPI `rknn-toolkit2==2.3.2` is the latest available RKNN converter package and its metadata still requires `numpy<=1.26.4`. Newer Linux MediaPipe/OpenCV-Contrib packages may prefer NumPy 2.x, so the correct fix is environment separation plus pinned requirements.

## RKNN Missing pkg_resources

Symptom:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

Cause:

`rknn-toolkit2==2.3.2` still imports `pkg_resources`, but newer `setuptools` versions removed it.

Fix:

```bash
python -m pip install -r requirements-rknn.txt
```

The RKNN requirements file pins `setuptools==80.9.0`.

## RKNN Noisy Non-Fatal Logs

Symptom:

```text
UserWarning: pkg_resources is deprecated as an API
W load_onnx: The config.mean_values is None
W load_onnx: The config.std_values is None
E RKNN: Unkown op target: 0
```

Meaning:

These messages are noisy but not necessarily fatal. In the current MindFace-Lite MLP export, the conversion is successful when the final report contains:

```json
{
  "exported": true
}
```

and the output file exists:

```text
outputs/models/mlp_mouth.rk3588.rknn
```

`pkg_resources` is handled by pinning `setuptools==80.9.0` in `requirements-rknn.txt`. The `mean_values/std_values` messages are expected because this project feeds normalized audio features, not images. `Unkown op target: 0` appears in RKNN-Toolkit2 logs for this tiny MLP conversion, but the build/export can still finish successfully.

To reduce RKNN debug output:

```yaml
runtime:
  verbose: false
```

Set it to `true` only when debugging RKNN conversion internals.

## RKNN Unavailable in Backend Consistency Report

Symptom in `outputs/reports/backend_consistency_report.json`:

```json
{
  "rknn": {
    "available": false
  }
}
```

Meaning:

This is expected in the Windows training environment if `rknn-toolkit2` is not installed or `outputs/models/mlp_mouth.rk3588.rknn` has not been generated.

Fix for real RKNN comparison:

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python scripts/24_rknn_convert_and_infer.py --config configs/rknn_deploy.yaml
python scripts/17_compare_inference_backends.py --config configs/consistency_compare.yaml
```

## MediaPipe Missing Linux GL Libraries

Symptom:

```text
OSError: libGLESv2.so.2: cannot open shared object file
```

Fix on a separate WSL optional-feature environment:

```bash
sudo apt install -y libgles2 libegl1
```

Do not install MediaPipe into `mindface-rknn`; keep RKNN conversion and MediaPipe landmark extraction in separate environments.

## MediaPipe Not Installed in mindface-rknn

Symptom:

```text
MediaPipe FaceLandmarker dependencies are unavailable.
Original error: No module named 'mediapipe'
```

Meaning:

This is expected in the WSL RKNN environment. `mindface-rknn` should stay focused on RKNN conversion, Device Tree, and cross-compilation. Installing `mediapipe` there can pull `opencv-contrib-python` and break the RKNN `numpy<=1.26.4` dependency.

Recommended fix on Windows:

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
conda activate mindface-lite
python -m pip install -r requirements-optional.txt
python scripts/14_extract_grid_video_landmarks.py --check-deps
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_debug
```

Alternative fix in a separate WSL environment:

```bash
python3 -m venv ~/.venvs/mindface-media
source ~/.venvs/mindface-media/bin/activate
python -m pip install --upgrade pip setuptools wheel
sudo apt install -y libgles2 libegl1 ffmpeg
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
python -m pip install -e .
python scripts/14_extract_grid_video_landmarks.py --check-deps
```

Do not reuse `~/.venvs/mindface-rknn` for this.

## MediaPipe GPU Delegate

MediaPipe Python Tasks expose `BaseOptions.Delegate.GPU`, and MindFace-Lite supports it through:

```yaml
landmarks:
  delegate: gpu
```

or:

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --max-videos 8 --output-dir data/processed/grid_video_landmarks_gpu_debug --delegate gpu
```

However, GPU delegate support is platform-dependent. If GPU delegate fails on Windows, use:

```yaml
landmarks:
  delegate: cpu
```

The CPU path uses MediaPipe/XNNPACK and is the stable default for the GRID preprocessing stage.

On Windows `mindface-lite`, the current MediaPipe pip package may fail with:

```text
GPU processing is disabled in build flags
```

This means the installed MediaPipe wheel was not built with GPU graph support. It is not an NVIDIA driver or PyTorch CUDA problem. Use CPU/XNNPACK for this stage, or test GPU delegate in a separate Ubuntu MediaPipe environment.

## OpenCV Missing VideoWriter

Symptom:

```text
AttributeError: module 'cv2' has no attribute 'VideoWriter'
```

Cause:

This can happen after uninstalling `opencv-contrib-python`; the shared `cv2` namespace may be left in a broken state.

Fix in `mindface-rknn` without changing `numpy`:

```bash
python -m pip install --force-reinstall --no-deps opencv-python==4.11.0.86
python -m pip check
```

## sounddevice Missing PortAudio

Symptom:

```text
OSError: PortAudio library not found
```

Fix on WSL/Linux:

```bash
sudo apt install -y portaudio19-dev libportaudio2
```

On Windows, install `sounddevice` in the `mindface-lite` conda environment and make sure a microphone device is available.

## sounddevice Error Querying Device -1

Symptom:

```text
sounddevice.PortAudioError: Error querying device -1
default.device=[-1, -1]
```

Cause:

PortAudio is installed, but it cannot see a default input device. This is common inside WSL because microphone input is not always exposed to Linux apps.

Check visible devices:

```bash
python scripts/23_mic_stream_rule_demo.py --list-devices
```

If no input devices are listed, use the Windows `mindface-lite` environment for microphone streaming:

```powershell
conda activate mindface-lite
python -m pip install -r requirements-optional.txt
python scripts/23_mic_stream_rule_demo.py --list-devices
```

Then set `configs/mic_stream.yaml`:

```yaml
audio:
  device: 1
```

Use the input device index/name shown by `--list-devices`. Keep `device: null` only when PortAudio has a valid default input device.

## pyttsx3 Not Installed

Symptom:

```text
RuntimeError: pyttsx3 is not installed
```

Fix in the Windows training environment:

```powershell
python -m pip install -r requirements-optional.txt
```

On WSL/Linux, `pyttsx3` may also need a local speech engine such as `espeak-ng`.

## pyttsx3 espeak Shutdown Crash

Symptom:

```text
Fatal Python error: _enter_buffered_busy
Exception ignored on calling ctypes callback function: EspeakDriver._onSynth
Aborted (core dumped)
```

Cause:

On Linux/WSL, `pyttsx3` uses the espeak driver. Some versions leave callback threads alive during Python interpreter shutdown, so the WAV may be created but the process exits with an abort.

Fix:

`configs/real_tts.yaml` uses:

```yaml
pyttsx3:
  isolate_subprocess: auto
```

This runs pyttsx3 generation in a child process on Linux. The main script accepts the WAV only if the output file exists and has valid content, so `scripts/22_real_tts_mouth_demo.py` can continue to generate CSV and MP4.

Recommended production path:

```powershell
conda activate mindface-lite
python scripts/21_real_tts_generate_wav.py --config configs/real_tts.yaml
```

Use `mindface-rknn` only for RKNN conversion, Device Tree, and cross-compilation work.

## Shape Mismatch

MLP expects `[frames, feature_dim]`.

LSTM, TCN, and Transformer expect `[batch, frames, feature_dim]`.

The model factory and inference helpers handle this automatically when the checkpoint metadata is correct.
