# Stage 0 and Stage 1: Rule-Based Mouth Demo

## What To Build

Stage 0 prepares the Python project structure, requirements, YAML config, logging, and output folders.

Stage 1 reads a WAV file, computes frame-level RMS audio energy, maps that energy to `mouth_open` in `[0, 1]`, draws a simple digital face with OpenCV, exports an MP4 video, and saves a CSV log.

Stage 1.5 keeps the same RMS rule logic but uses a more polished OpenCV renderer for demos and interviews.

Stage 1.6 keeps the same RMS rule logic but uses a generated static face image and OpenCV mouth-region warping.

## Why It Matters

This is the smallest complete loop in the project:

```text
audio input -> feature extraction -> control parameter -> visualization -> validation files
```

It is not machine learning yet, but it builds the debugging foundation for later ML models.

## Knowledge Involved

- WAV format: sample rate, channels, 16-bit PCM samples.
- Frame processing: convert continuous audio into video-rate frames.
- RMS energy: a simple loudness feature.
- Normalization: map raw energy to a bounded control signal.
- Smoothing: reduce jitter for a more stable mouth motion.
- OpenCV: draw frames and encode MP4 video.
- CSV logging: inspect per-frame values.

## Folder Structure

```text
configs/demos/rule-demo.yaml
scripts/00_generate_test_audio.py
scripts/01_rule_mouth_demo.py
scripts/01_5_better_visual_demo.py
scripts/01_6_expressive_avatar_demo.py
src/mindface/audio/features.py
src/mindface/visual/mouth_drawer.py
src/mindface/visual/better_mouth_drawer.py
src/mindface/visual/static_avatar_warper.py
src/mindface/utils/logger.py
assets/avatar/stage1_6_static_face.png
outputs/audio/
outputs/videos/
outputs/logs/
```

## Runnable Commands

```powershell
python scripts/00_generate_test_audio.py
python scripts/01_rule_mouth_demo.py --config configs/demos/rule-demo.yaml
python scripts/01_5_better_visual_demo.py --config configs/demos/better-visual-demo.yaml
python scripts/01_6_expressive_avatar_demo.py --config configs/demos/expressive-avatar-demo.yaml
```

## Expected Output

```text
outputs/audio/test_voice.wav
outputs/videos/rule_mouth_demo.mp4
outputs/videos/better_visual_mouth_demo.mp4
outputs/videos/expressive_avatar_demo.mp4
outputs/videos/expressive_avatar_preview.png
outputs/logs/rule_demo.csv
outputs/logs/better_visual_rule_demo.csv
outputs/logs/expressive_avatar_demo.csv
```

The CSV contains:

```text
frame_index,time_sec,rms,mouth_open
```

## RMS Formula

For one audio frame with `N` samples:

```text
RMS = sqrt((x_0^2 + x_1^2 + ... + x_(N-1)^2) / N)
```

If the sound is louder, sample amplitudes are larger, squared values are larger, and RMS becomes larger.

## How `mouth_open` Is Computed

1. Compute RMS per video frame.
2. Subtract a small `noise_floor` so background noise does not open the mouth.
3. Normalize by the 95th percentile of RMS energy.
4. Clip to `[0, 1]`.
5. Apply `gamma` to shape the response curve.
6. Apply exponential smoothing to reduce jitter.

The final value is:

```text
mouth_open = smooth(clip((rms - noise_floor) / scale, 0, 1) ^ gamma)
```

## Common Errors And Fixes

- Missing WAV file: run `python scripts/00_generate_test_audio.py`.
- OpenCV cannot write MP4: reinstall `opencv-python` or try a shorter output path.
- CSV is all zeros: check whether the WAV is silent or `noise_floor` is too high.
- Mouth is always open: lower the audio amplitude or raise `noise_floor`.
- Video exists but will not play: use a standard player that supports MP4 `mp4v`.

## Stage 1.5 Better Visual Renderer

Stage 1.5 is not a new algorithm. It is a better visualization layer:

```text
same WAV -> same RMS -> same mouth_open -> better OpenCV face renderer
```

The purpose is to keep Stage 1 easy to debug while also having a cleaner video for demonstration. The new renderer removes the default debug text, adds face shading, eyes, eyebrows, nose, lips, teeth and tongue. It still uses only OpenCV drawing primitives, so it remains deterministic and easy to run.

## Stage 1.6 Expressive Static Avatar

Stage 1.6 uses a generated static face image:

```text
assets/avatar/stage1_6_static_face.png
```

Then it applies local mouth ROI deformation:

```text
same WAV -> same RMS -> same mouth_open -> OpenCV remap on mouth region
```

The mouth region is configured in `configs/demos/expressive-avatar-demo.yaml` with normalized coordinates. The renderer moves upper and lower lip pixels apart as `mouth_open` increases, then overlays the dark mouth cavity, teeth and tongue. This is closer to a real digital-human rendering pipeline than the fully drawn Stage 1.5 face, but it is still lightweight and educational.

## Interview Explanation

This stage demonstrates that I can build a complete signal-processing demo before adding ML. I take raw audio, compute a meaningful feature, map it to a bounded control parameter, render a visual result, and export logs for debugging. This is the same engineering pattern used later for model inference: replace the rule function with a trained neural network, keep the input/output validation pipeline.
