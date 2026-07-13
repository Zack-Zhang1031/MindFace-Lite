# Expressive Static Avatar

## overview

Stage 1.6 adds a static-image avatar renderer. The project uses a generated front-facing face image as a source asset, then applies OpenCV local mouth-region mesh deformation according to RMS-derived `mouth_open` plus simple viseme controls.

## design decisions

- Keep Stage 1 as the transparent signal-processing baseline.
- Keep Stage 1.5 as a fully code-drawn 2D renderer.
- Add Stage 1.6 to demonstrate a more realistic digital-human direction: a static face image plus local mouth mesh warping.
- Store the generated image prompt in `assets/avatar/stage1_6_static_face.prompt.txt` so the asset is explainable and reproducible at the project level.
- Use configurable normalized mouth ROI coordinates, so the same code can work with another static face image after tuning YAML values.
- Use a small local control mesh instead of a single global ROI remap. This keeps the mouth corners, upper lip, lower lip, and center area independently movable.
- Keep viseme logic rule-based in this stage. It is not a phoneme recognizer yet; it is a transparent demo for how `mouth_width` and `lip_round` affect visual style.

## implementation notes

Run:

```powershell
python scripts/00_generate_test_audio.py
python scripts/01_6_expressive_avatar_demo.py --config configs/demos/expressive-avatar-demo.yaml
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

The renderer lives in `src/mindface/visual/static_avatar_warper.py`. It extracts the mouth ROI, applies piecewise affine warping on local control points, blends the warped region back with a soft feather mask, then overlays mouth cavity, teeth, tongue, cheek pulse, blink, and light head motion.

The Stage 1.6 CSV records:

```text
frame_index,time_sec,rms,mouth_open,mouth_width,lip_round,viseme
```

The demo maps simple viseme labels as follows:

| viseme | Effect |
| --- | --- |
| `closed` | Forces `mouth_open` close to zero for m/b/p-style closure. |
| `a` | Large vertical mouth opening, similar to “啊”. |
| `i` | Wider and flatter mouth, similar to “一”. |
| `o` | Rounded medium opening, similar to “哦”. |
| `u` | Narrow rounded lips, similar to “乌”. |

This stage is still a visual renderer, not a real lip-reading or phoneme alignment model. Later ML stages should learn these parameters from audio features and landmark/blendshape labels.
