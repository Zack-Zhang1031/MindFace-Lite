# Better Visual Renderer

## overview

Stage 1.5 adds a more polished OpenCV avatar renderer on top of the Stage 1 RMS rule pipeline. It keeps the same audio feature and `mouth_open` control logic, but renders a cleaner 2D face with face shading, eyes, eyebrows, nose, lips, teeth and tongue.

## design decisions

- Keep Stage 1 simple renderer as the debugging baseline.
- Add Stage 1.5 as a separate renderer and output path, so visual polish does not hide signal-processing logic.
- Avoid external image assets in the first better renderer, keeping the demo deterministic and easy to run.
- Use OpenCV drawing primitives only, because the project already depends on OpenCV.
- Keep `mouth_open` as the main control parameter, while accepting optional `mouth_width` and `lip_round` for future model outputs.

## implementation notes

Run:

```powershell
python scripts/00_generate_test_audio.py
python scripts/01_5_better_visual_demo.py --config configs/demos/better-visual-demo.yaml
```

Outputs:

```text
outputs/videos/better_visual_mouth_demo.mp4
outputs/logs/better_visual_rule_demo.csv
outputs/logs/better_visual_demo.log
```

The renderer lives in `src/mindface/visual/better_mouth_drawer.py`. It is still not a production digital human renderer; it is a better educational visualization for demos and interviews.

