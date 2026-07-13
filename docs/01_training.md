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
python scripts/02_generate_synthetic_dataset.py --config configs/datasets/synthetic-dataset.yaml
python scripts/03_train_model.py --config configs/training/train-mlp.yaml
```

Optional models:

```powershell
python scripts/03_train_model.py --config configs/training/train-lstm.yaml
python scripts/03_train_model.py --config configs/training/train-tcn.yaml
python scripts/03_train_model.py --config configs/training/train-transformer.yaml
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
python scripts/09_prepare_grid_dataset.py --config configs/datasets/prepare-grid.yaml
```

Then train:

```powershell
python scripts/03_train_model.py --config configs/training/train-grid-mlp.yaml
```

The current GRID preprocessing uses RMS-based pseudo labels. It validates the training pipeline on real GRID audio, but true lip-sync labels require a later video landmark extraction stage.

## Key Files

- `src/mindface/data/synthetic_dataset.py`: frame or sequence dataset.
- `src/mindface/models/mlp.py`: simple baseline.
- `src/mindface/models/lstm.py`: sequence recurrent baseline.
- `src/mindface/models/tcn.py`: temporal convolution model.
- `src/mindface/models/transformer.py`: Transformer encoder model.
- `src/mindface/training/trainer.py`: training and validation loop.

## Model Bundle And Resume

Training uses one `FeatureSpec` for dataset extraction, checkpoint metadata, inference and ONNX export. The best checkpoint stores the best validation model, while `output.last_checkpoint_path` is updated after every epoch and also stores optimizer state and epoch.

To resume, set:

```yaml
train:
  epochs: 6
  resume_from: outputs/checkpoints/mlp_mouth.last.pt
```

`epochs` is the final target epoch, not the number of additional epochs. Resume validates model type, model parameters and FeatureSpec before loading optimizer state.

## Interview Explanation

The MLP baseline proves the data path and loss are correct. LSTM, TCN, and Transformer introduce temporal context. In a real lip-sync project, the labels would come from facial landmarks or blendshape extraction. The same training loop would remain useful.
