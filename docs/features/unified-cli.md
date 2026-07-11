# Unified CLI

## overview

MindFace-Lite now provides a small unified CLI named `mindface`. It wraps the most common script entry points while keeping every original script runnable from the project root.

## design decisions

- Keep scripts as the source of truth because this is an educational project and individual stages should remain easy to inspect.
- Add `python -m mindface ...` and the installed `mindface ...` command for daily use.
- Avoid hiding YAML configs. CLI commands still call the same config-driven scripts.
- Keep subcommands narrow and predictable instead of building a large framework.

## implementation notes

Install editable mode once:

```powershell
cd C:\Users\Administrator\Desktop\MindFace-Lite
conda activate mindface-lite
python -m pip install -e .
```

Common commands:

```powershell
python -m mindface health
python -m mindface rule-demo
python -m mindface better-visual
python -m mindface expressive-avatar
python -m mindface train --config configs/train_mlp.yaml
python -m mindface export-onnx --config configs/export_onnx.yaml
python -m mindface compare-backends
python -m mindface prepare-grid-landmark --config configs/prepare_grid_landmark.yaml --max-samples 8
```

If editable install has registered the console script, the shorter form also works:

```powershell
mindface health
mindface expressive-avatar
```

The CLI implementation is intentionally thin:

```text
src/mindface/cli.py
src/mindface/__main__.py
```

For debugging, run the underlying `scripts/*.py` file directly so the exact stage remains visible.
