# GRID Preprocessing

## Why Raw GRID Cannot Be Trained Directly

The trainer expects:

```text
manifest.csv
features_000000.npy
targets_000000.npy
...
```

Raw GRID contains:

```text
data/raw/grid/audio/<speaker>/<utterance>.wav
data/raw/grid/alignments/<speaker>/<utterance>.align
data/raw/grid/video/<utterance>.mpg
```

So a preprocessing step is required before training.

## Current Preprocessing Script

Use:

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/datasets/prepare-grid.yaml
```

This scans:

```text
data/raw/grid/audio
data/raw/grid/alignments
data/raw/grid/video
```

Then writes:

```text
data/processed/grid_mouth/manifest.csv
data/processed/grid_mouth/features_000000.npy
data/processed/grid_mouth/targets_000000.npy
...
```

## Quick Debug Run

For a small test:

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/datasets/prepare-grid.yaml --max-samples 8 --output-dir data/processed/grid_mouth_debug
```

## Full GRID Audio Run

For all available GRID audio samples:

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/datasets/prepare-grid.yaml
```

The current config has:

```yaml
scan:
  require_alignment: true
  require_video: false
```

That means it can process the full audio/alignment set even though only part of the dataset has video files.

## Train After Preprocessing

After full preprocessing:

```powershell
python scripts/03_train_model.py --config configs/training/train-grid-mlp.yaml
```

## Important Label Boundary

The current GRID preprocessing creates pseudo labels:

```text
audio waveform -> RMS -> mouth_open -> mouth_width/lip_round
```

This is useful for testing the full training pipeline on real GRID audio, but it is not true video-derived lip motion.

True lip-sync training requires video label extraction:

```text
GRID video -> mouth landmarks / blendshape labels -> targets_*.npy
```

MindFace-Lite now provides the MediaPipe landmark extraction and audio/target alignment entry points.

## Landmark Supervised Dataset

After GRID video landmark extraction:

```powershell
python scripts/14_extract_grid_video_landmarks.py --config configs/datasets/grid-video-landmarks.yaml
```

Build a training manifest that uses landmark-derived targets instead of RMS pseudo labels:

```powershell
python scripts/16_prepare_grid_landmark_dataset.py --config configs/datasets/prepare-grid-landmark.yaml
```

Debug subset:

```powershell
python scripts/16_prepare_grid_landmark_dataset.py --config configs/datasets/prepare-grid-landmark.yaml --max-samples 8
```

Then train:

```powershell
python scripts/03_train_model.py --config configs/training/train-grid-landmark-mlp.yaml
```

This path expects:

```text
data/raw/grid/audio
data/processed/grid_video_landmarks/manifest.csv
data/processed/grid_video_landmarks/targets_*.npy
```

It writes:

```text
data/processed/grid_landmark_mouth/manifest.csv
data/processed/grid_landmark_mouth/features_*.npy
data/processed/grid_landmark_mouth/targets_*.npy
```

The training target columns are:

```text
mouth_open
mouth_width
mouth_round
```

If `data/raw/grid` is missing, this training path cannot run against real data yet.

## Manifest Columns

```text
schema_version
split
split_strategy
features
targets
speaker
utterance
frames
sample_rate
label_source
source_audio
source_alignment
source_video
```

The trainer only needs `split`, `features`, and `targets`. The other columns are for traceability and debugging.

GRID audio defaults to `speaker_disjoint`, so every speaker belongs to exactly one split. Video landmark extraction uses the same strategy when videos are grouped under speaker directories; for flat video directories it records `sample_fallback` because speaker identity cannot be recovered safely from an utterance name alone.
