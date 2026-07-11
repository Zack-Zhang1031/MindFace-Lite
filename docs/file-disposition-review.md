# File Disposition Review

This review separates the current MindFace-Lite mainline from legacy, external, or generated files.

Status after cleanup:

- Kept: `data/raw/grid`, `cpp`, and `tools`.
- Archived: legacy package/configs/scripts/docs into `archive/legacy-2026-07-06/`.
- Cleaned: `build`, `__pycache__`, old synthetic NPZ files, old 4-digit synthetic WAV files, and duplicate old root-level outputs.
- Archived root-level ONNX debug files into `archive/root-onnx-debug-2026-07-10/`.

## Can The Current Training Code Train `data/raw/grid` Directly?

Not directly. The current training code trains any manifest-formatted dataset, including synthetic data, GRID RMS pseudo labels, and GRID video-landmark labels. Raw GRID still needs preprocessing first.

```text
data/synthetic_mouth/manifest.csv
data/processed/grid_mouth/manifest.csv
data/processed/grid_landmark_mouth/manifest.csv
```

Expected training format:

```text
manifest.csv rows -> split, features, targets, wav
features_XXX.npy  -> [frames, 70] audio features
targets_XXX.npy   -> [frames, 3] mouth parameters
```

`data/raw/grid` is raw corpus data. It needs one of these preprocessing stages:

```text
raw GRID audio/alignment
  -> scripts/09_prepare_grid_dataset.py
  -> RMS pseudo mouth labels
  -> data/processed/grid_mouth/manifest.csv

raw GRID audio/video + extracted MediaPipe landmarks
  -> scripts/14_extract_grid_video_landmarks.py
  -> scripts/16_prepare_grid_landmark_dataset.py
  -> video-landmark mouth labels
  -> data/processed/grid_landmark_mouth/manifest.csv
```

Train with:

```powershell
python scripts/03_train_model.py --config configs/train_grid_mlp.yaml
python scripts/03_train_model.py --config configs/train_grid_landmark_mlp.yaml
```

## Raw GRID Scan

```text
data/raw/grid total: 69010 files, about 6309 MB
alignments: 34000 .align files, about 4.36 MB
audio:      34000 .wav files, about 2982.93 MB
video:       1000 .mpg files, about 411.37 MB
archives:       3 .zip files, about 2909.87 MB
```

Decision kept:

- Keep in place for future GRID preprocessing.
- It can still be moved outside the project later if disk usage or scan speed becomes a problem.

## Legacy Or Pre-Existing Source-Like Content

These were present before the current `src/mindface` mainline was created, or are not part of the new mainline.

```text
src/mindface_lite/                         legacy Python package, 20 source files
configs/00_rule_demo.yaml                  old numbered config
configs/01_generate_dataset.yaml           old numbered config
configs/02_train_mlp.yaml                  old numbered config
configs/03_infer_pytorch.yaml              old numbered config
configs/04_export_onnx.yaml                old numbered config
configs/05_infer_onnx.yaml                 old numbered config
configs/06_realtime_rule.yaml              old numbered config
configs/07_benchmark.yaml                  old numbered config
configs/demo_rule.yaml                     old demo config
configs/grid_paths.yaml                    GRID helper config
scripts/10_rknn_convert.py                 legacy RKNN helper
scripts/11_rknn_sim_infer_to_csv.py        legacy RKNN helper
scripts/12_udp_receiver.py                 legacy UDP helper
scripts/13_compare_outputs.py              legacy compare helper
scripts/14_inspect_onnx.py                 legacy ONNX helper
scripts/16_check_target_arch.sh            legacy embedded helper
scripts/17_collect_board_info.sh           legacy embedded helper
scripts/18_cross_compile_cpp.sh            legacy embedded helper
scripts/21_realtime_tts_mouth_demo.py      legacy TTS/realtime helper
scripts/22_prepare_grid_dirs.py            legacy GRID helper
scripts/23_grid_inspect_structure.py       legacy GRID helper
scripts/run_01_tts_pipeline.py             legacy aggregate runner
scripts/run_02_grid_prepare.py             legacy aggregate runner
scripts/run_03_edge_simulation_guide.py    legacy aggregate runner
scripts/run_04_rknn_simulation_guide.py    legacy aggregate runner
docs/00_FILE_MERGE_PLAN.md                 legacy merge note
docs/08_grid_dataset_next_step.md          legacy GRID note
docs/09_cross_compile_device_tree_uboot.md legacy embedded note
docs/10_rk3588_board_deploy_checklist.md   legacy RK3588 note
docs/11_interview_embedded_answer.md       legacy interview note
docs/12_tts_realtime_drive.md              legacy TTS note
docs/13_model_knowledge_map.md             legacy model note
docs/14_project_coverage_checklist.md      legacy coverage note
tools/                                     legacy toolchain/device-tree helpers
```

Decision applied:

- Moved to `archive/legacy-2026-07-06/` and kept as reference.

## Data Artifacts Not Used By The New Mainline

```text
data/sample_voice_like.wav
data/synthetic_mouth/features/*.npz        80 old NPZ feature files
data/synthetic_mouth/wavs/sample_0000.wav  old 4-digit WAV naming pattern
...
data/synthetic_mouth/wavs/sample_0079.wav  80 old 4-digit WAV files
```

The current generated dataset uses:

```text
data/synthetic_mouth/manifest.csv
data/synthetic_mouth/features_000.npy ... features_039.npy
data/synthetic_mouth/targets_000.npy ... targets_039.npy
data/synthetic_mouth/wavs/sample_000.wav ... sample_039.wav
```

Decision applied:

- Deleted old NPZ artifacts and old 4-digit WAV artifacts.
- Kept the current manifest-driven synthetic dataset.

## Generated Artifacts That Are Safe To Recreate

```text
build/
outputs/
**/__pycache__/
```

Decision applied:

- Deleted `build/` and `__pycache__/`.
- Deleted duplicate old root-level outputs.
- Moved root-level ONNX debug files `check0_base_optimize.onnx`, `check2_correct_ops.onnx`, and `check3_fuse_ops.onnx` to `archive/root-onnx-debug-2026-07-10/`.
- Kept current demo outputs under `outputs/audio`, `outputs/videos`, `outputs/logs`, `outputs/models`, `outputs/checkpoints`, and `outputs/reports`.

## Recommended Cleanup Plan

For a clean educational repo:

1. Keep `src/mindface`, current `configs/*.yaml`, current `scripts/00-08`, docs `00-08`, README, C++ demos, and requirements.
2. Move legacy code/config/docs to `archive/legacy-2026-07-06/`.
3. Move `data/raw/grid` outside the repo or keep only a small subset until GRID preprocessing is implemented.
4. Delete `build`, `__pycache__`, and old synthetic NPZ/4-digit WAV artifacts when you no longer need demo outputs.
