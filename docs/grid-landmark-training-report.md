# GRID Landmark Training Report

## Status

当前报告是模板。真实 GRID landmark 训练需要先恢复：

```text
data/raw/grid/audio
data/raw/grid/video
data/processed/grid_video_landmarks/manifest.csv
```

数据恢复后，用本文件记录真实训练结果，不要把 GRID 原始数据、模型权重或大体积输出提交到 GitHub。

## Reproduction Commands

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
conda activate mindface-lite

python scripts/14_extract_grid_video_landmarks.py --config configs/grid_video_landmarks.yaml
python scripts/16_prepare_grid_landmark_dataset.py --config configs/prepare_grid_landmark.yaml
python scripts/03_train_model.py --config configs/train_grid_landmark_mlp.yaml
python scripts/05_export_onnx.py --config configs/export_onnx.yaml
python scripts/17_compare_inference_backends.py --config configs/consistency_compare.yaml
```

## Dataset Summary

| Item | Value |
| --- | --- |
| Raw GRID audio path | `data/raw/grid/audio` |
| Raw GRID video path | `data/raw/grid/video` |
| Landmark manifest | `data/processed/grid_video_landmarks/manifest.csv` |
| Supervised manifest | `data/processed/grid_landmark_mouth/manifest.csv` |
| Train samples | TODO |
| Val samples | TODO |
| Test samples | TODO |
| Total aligned frames | TODO |
| Mean face detection rate | TODO |
| Low-detection samples | TODO |

## Training Result

| Metric | Value |
| --- | --- |
| Model | `mlp` |
| Checkpoint | `outputs/checkpoints/grid_landmark_mlp_mouth.pt` |
| Epochs | TODO |
| Best val loss | TODO |
| Train duration | TODO |
| CUDA device | TODO |

Loss curve source:

```text
outputs/experiments/<timestamp>_train_mlp/history.csv
```

## Backend Consistency

Report source:

```text
outputs/reports/backend_consistency_report.json
```

| Backend | Available | MAE vs PyTorch | RMSE vs PyTorch | Max abs error | Latency |
| --- | --- | --- | --- | --- | --- |
| PyTorch | TODO | Reference | Reference | Reference | TODO |
| ONNXRuntime | TODO | TODO | TODO | TODO | TODO |
| RKNN | TODO | TODO | TODO | TODO | TODO |

## Notes

- `09_prepare_grid_dataset.py` uses RMS pseudo labels and is useful for pipeline debugging.
- `16_prepare_grid_landmark_dataset.py` uses video landmark targets and is the real supervised training path.
- RKNN consistency requires WSL `mindface-rknn` or an RK3588 runtime target.
