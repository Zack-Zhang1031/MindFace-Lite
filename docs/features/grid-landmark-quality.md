# GRID Landmark 标签质量报告

## overview

GRID 视频 landmark 提取后会生成质量报告，用于判断 MediaPipe 检测是否稳定、标签分布是否合理、哪些样本需要人工复查。

## design decisions

- 质量报告从已生成的 `manifest.csv` 和 `landmarks/*.csv` 计算，不需要重复跑 MediaPipe。
- 默认低检测阈值为 `0.95`，可在 `configs/grid_video_landmarks.yaml` 中配置。
- 报告只评估 landmark 标签提取质量，不声称音视频口型已经严格对齐。

## implementation notes

提取完成后自动生成：

```text
outputs/reports/grid_video_landmark_quality.json
```

也可以对已有结果单独生成：

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml --quality-only
```

报告包含整体检测率、每个样本检测率、`mouth_open/mouth_width/mouth_round` 分布和低检测样本列表。

