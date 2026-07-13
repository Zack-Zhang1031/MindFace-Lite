# RK3588 and RKNN Deployment Plan

## Deployment Path

```text
PyTorch checkpoint -> ONNX -> RKNN conversion -> INT8 calibration -> RK3588 runtime test
```

## Current Project Scripts

```powershell
python scripts/24_rknn_convert_and_infer.py --check-deps
python scripts/24_rknn_convert_and_infer.py --config configs/deployment/rknn-deploy.yaml
python scripts/25_device_tree_uboot_check.py --check-deps
python scripts/25_device_tree_uboot_check.py --config configs/deployment/device-tree-uboot.yaml
```

Implemented files:

- `src/mindface/deploy/rknn_tools.py`
- `configs/deployment/rknn-deploy.yaml`
- `scripts/24_rknn_convert_and_infer.py`
- `tools/embedded/rk3588-mouth-uart-overlay.dts`
- `configs/deployment/device-tree-uboot.yaml`
- `scripts/25_device_tree_uboot_check.py`

The RKNN script is real conversion code, but it still requires Rockchip's RKNN development environment. The Device Tree script can compile an overlay when `dtc` is available and prints U-Boot apply commands.

## RK3588 Concerns

- NPU operator support: avoid unsupported or dynamic operators when possible.
- DDR bandwidth: audio features are small, but video or landmark models can become bandwidth-heavy.
- Power and heat: sustained realtime inference needs thermal planning.
- Interface limits: microphone, camera, UART, USB, Ethernet, and actuator timing matter.
- CPU/NPU scheduling: keep audio capture and control responsive while NPU runs inference.

## INT8 Calibration

Calibration needs representative feature tensors. For this project, calibration samples should come from the same audio feature extractor used at runtime.

Bad calibration symptoms:

- `mouth_open` becomes nearly constant.
- Small speech sounds disappear.
- Loud sounds clip to 1.0 too often.
- ONNX and RKNN outputs diverge.

## Device Tree, U-Boot, Kernel

This repository documents the concepts but does not modify a real board. For hardware collaboration, the important questions are:

- Which UART, I2S, USB, or GPIO interfaces are used?
- Are clocks, pinmux, and power rails configured?
- Does the kernel expose the device node needed by the control program?
- Is thermal throttling affecting inference latency?

## Interview Explanation

I understand the model conversion path and the hardware constraints that affect deployment. I would coordinate with hardware and BSP engineers around interfaces, power, heat, kernel drivers, and NPU runtime constraints.
