# 环境矩阵

## overview

MindFace-Lite 使用两个主要环境：Windows `mindface-lite` 负责训练和多媒体功能，WSL/Ubuntu `mindface-rknn` 负责 RKNN 转换、Device Tree Compiler 和交叉编译工具链。

## design decisions

- 不把 MediaPipe、CUDA PyTorch、RKNN-Toolkit2 混装在同一个环境。
- Windows 环境优先保证 PyTorch、ONNXRuntime、MediaPipe、TTS、麦克风和 C++ demo 可用。
- WSL 环境优先保证 RKNN-Toolkit2 的 NumPy/ONNX 兼容性，以及 Linux 嵌入式工具链可用。

## implementation notes

| 脚本范围 | 推荐环境 | 原因 |
| --- | --- | --- |
| `00` 到 `13` | Windows `mindface-lite` | 训练、推理、ONNX、量化、剪枝 |
| `14` | Windows `mindface-lite` | MediaPipe Face Landmarker 在 Windows 环境已验证 |
| `19` 到 `23` | Windows `mindface-lite` | TTS、麦克风、OpenCV 预览更稳定 |
| `24` | WSL `mindface-rknn` | RKNN-Toolkit2 面向 Ubuntu x86_64 转换环境 |
| `25` | WSL `mindface-rknn` | `dtc` 和 ARM64 交叉编译工具链属于 Linux/BSP 侧 |
| `cpp/` Windows demo | Windows | 生成 `.exe` 并验证 UDP/串口式输出 |

