# MindFace-Lite 项目演示稿

## 1. 项目一句话

MindFace-Lite 是一个教育型 AI 工程项目：从音频 RMS 规则口型 demo 出发，逐步扩展到 PyTorch 训练、ONNX/RKNN 部署、量化剪枝、实时输入、C++ 控制和 RK3588 边缘部署思考。

## 2. 推荐演示环境

Windows 训练环境：

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
conda activate mindface-lite
python -m pip check
```

WSL RKNN 环境只负责 RKNN、Device Tree、交叉编译：

```bash
cd /mnt/c/Users/Administrator/Desktop/MindFace-Lite
source ~/.venvs/mindface-rknn/bin/activate
python -m pip check
```

## 3. 最短演示路线

```powershell
mindface ui
mindface health
mindface config validate --all
mindface pipeline basic --dry-run
mindface demo generate-audio
mindface demo rule
mindface demo better-visual
mindface demo expressive-avatar
mindface train --config configs/training/train-mlp.yaml
mindface export onnx
mindface benchmark backends
```

重点查看：

```text
outputs/videos/rule_mouth_demo.mp4
outputs/videos/better_visual_mouth_demo.mp4
outputs/videos/expressive_avatar_demo.mp4
outputs/videos/expressive_avatar_preview.png
outputs/checkpoints/mlp_mouth.pt
outputs/models/mlp_mouth.onnx
outputs/reports/backend_consistency_report.json
```

## 4. Stage 1.6 视觉演示怎么讲

`mindface demo expressive-avatar` 使用一张静态人脸图，通过 OpenCV 对嘴部局部 mesh 做形变，核心实现位于 `src/mindface/visual/expressive_avatar.py`。

它不再只是整体拉伸 ROI，而是用局部控制点影响嘴角、上唇、下唇和口腔中心区域；边缘用 feather mask 混合回原图，避免硬边。配置里的 viseme 事件模拟：

```text
closed -> m/b/p 闭嘴
i      -> “一” 横向拉宽
a      -> “啊” 大张口
o      -> “哦” 圆唇
u      -> “乌” 更窄更圆
```

输出 CSV 包含：

```text
frame_index,time_sec,rms,mouth_open,mouth_width,lip_round,viseme
```

这一步的意义是把口型从单一 `mouth_open` 扩展成更接近数字人控制参数的三维口型空间。

## 5. GRID landmark 真实标签训练路线

需要真实 GRID 数据存在：

```text
data/raw/grid/audio
data/raw/grid/video
```

先提取视频 landmark 标签：

```powershell
mindface data extract-landmarks --check-deps
mindface data extract-landmarks --config configs/datasets/grid-video-landmarks.yaml
```

再把视频 landmark target 和 WAV 音频特征对齐成训练 manifest：

```powershell
mindface data align-landmarks --config configs/datasets/prepare-grid-landmark.yaml
```

最后训练真正以 landmark 标签为监督的 MLP：

```powershell
mindface train --config configs/training/train-grid-landmark-mlp.yaml
```

输出：

```text
data/processed/grid_landmark_mouth/manifest.csv
outputs/checkpoints/grid_landmark_mlp_mouth.pt
outputs/experiments/<timestamp>_train_mlp/
```

注意：如果当前项目目录没有 `data/raw/grid`，这一步不能真实训练，只能验证 CLI 入口和错误提示。

## 6. 后端一致性演示怎么讲

运行：

```powershell
mindface benchmark backends --config configs/benchmarks/backend-consistency.yaml
```

报告：

```text
outputs/reports/backend_consistency_report.json
```

讲解重点：

- PyTorch 是训练基准。
- ONNXRuntime 用同一段 audio features 做推理，和 PyTorch 比 MAE、RMSE、最大绝对误差。
- RKNN 是可选后端；Windows 环境通常只记录不可用原因，WSL/RKNN 或 RK3588 板端才能做真实 RKNN 数值对比。

## 7. 面试表达

可以这样总结：

“我把一个音频驱动口型问题拆成了完整工程链路：先用 RMS 做可解释 baseline，再构造数据集和 PyTorch 训练闭环，随后导出 ONNX、做 ONNXRuntime 推理、量化和剪枝 benchmark，最后扩展到麦克风实时输入、C++ 控制侧、RKNN/RK3588 部署思路。这个项目重点不是炫酷视觉，而是能讲清楚数据、模型、部署、调试、性能和边缘约束。”
