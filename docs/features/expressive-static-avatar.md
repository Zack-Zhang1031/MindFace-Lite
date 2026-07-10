# Expressive Static Avatar

## overview

Stage 1.6 adds a static-image avatar renderer. The project uses a generated front-facing face image as a source asset, then applies OpenCV local mouth-region deformation according to the same RMS-derived `mouth_open` signal.

## design decisions

- Keep Stage 1 as the transparent signal-processing baseline.
- Keep Stage 1.5 as a fully code-drawn 2D renderer.
- Add Stage 1.6 to demonstrate a more realistic digital-human direction: a static face image plus mouth ROI warping.
- Store the generated image prompt in `assets/avatar/stage1_6_static_face.prompt.txt` so the asset is explainable and reproducible at the project level.
- Use configurable normalized mouth ROI coordinates, so the same code can work with another static face image after tuning YAML values.

## implementation notes

Run:

```powershell
python scripts/00_generate_test_audio.py
python scripts/01_6_expressive_avatar_demo.py --config configs/expressive_avatar_demo.yaml
```

Inputs:

```text
assets/avatar/stage1_6_static_face.png
outputs/audio/test_voice.wav
```

Outputs:

```text
outputs/videos/expressive_avatar_demo.mp4
outputs/videos/expressive_avatar_preview.png
outputs/logs/expressive_avatar_demo.csv
outputs/logs/expressive_avatar_demo.log
```

The renderer lives in `src/mindface/visual/static_avatar_warper.py`. It extracts the mouth ROI, applies local `cv2.remap()` deformation to separate the upper/lower lip region, then overlays a dark mouth opening, teeth and tongue as `mouth_open` increases.

