# PyTorch Training Stage

## What To Build

The project generates a synthetic dataset of audio features and mouth parameters, then trains a model to predict:

```text
[mouth_open, mouth_width, lip_round]
```

Supported model types:

- `mlp`: frame-wise baseline.
- `lstm`: sequence model with recurrent memory.
- `tcn`: convolutional temporal model.
- `transformer`: self-attention sequence model.

## Why It Matters

An AI algorithm engineer should be able to complete the full loop:

```text
data -> Dataset -> DataLoader -> model -> loss -> optimizer -> train -> validate -> save -> load -> infer
```

This project keeps the first dataset synthetic so the code can be debugged before adding real lip-sync data.

## Commands

```powershell
python scripts/02_generate_synthetic_dataset.py --config configs/synthetic_dataset.yaml
python scripts/03_train_model.py --config configs/train_mlp.yaml
```

Optional models:

```powershell
python scripts/03_train_model.py --config configs/train_lstm.yaml
python scripts/03_train_model.py --config configs/train_tcn.yaml
python scripts/03_train_model.py --config configs/train_transformer.yaml
```

Every training run now writes an experiment trace:

```text
outputs/experiments/<timestamp>_train_<model>/
├── config.yaml
├── history.csv
└── metrics.json
outputs/experiments/latest_train_run.txt
```

`config.yaml` is the exact config snapshot for the run. `history.csv` stores epoch-level train/val loss. `metrics.json` stores dataset size, best validation loss, parameter count, device, runtime, PyTorch version and CUDA state.

## Train On Preprocessed GRID Audio

Raw GRID cannot be trained directly. First create a processed manifest:

```powershell
python scripts/09_prepare_grid_dataset.py --config configs/prepare_grid.yaml
```

Then train:

```powershell
python scripts/03_train_model.py --config configs/train_grid_mlp.yaml
```

The current GRID preprocessing uses RMS-based pseudo labels. It validates the training pipeline on real GRID audio, but true lip-sync labels require a later video landmark extraction stage.

## Key Files

- `src/mindface/data/synthetic_dataset.py`: frame or sequence dataset.
- `src/mindface/models/mlp.py`: simple baseline.
- `src/mindface/models/lstm.py`: sequence recurrent baseline.
- `src/mindface/models/tcn.py`: temporal convolution model.
- `src/mindface/models/transformer.py`: Transformer encoder model.
- `src/mindface/training/trainer.py`: training and validation loop.

## Interview Explanation

The MLP baseline proves the data path and loss are correct. LSTM, TCN, and Transformer introduce temporal context. In a real lip-sync project, the labels would come from facial landmarks or blendshape extraction. The same training loop would remain useful.
