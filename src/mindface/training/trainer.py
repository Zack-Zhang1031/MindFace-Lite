from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mindface.data.synthetic_dataset import MouthParameterDataset
from mindface.experiments.tracking import (
    base_runtime_info,
    make_experiment_dir,
    write_config_snapshot,
    write_history_csv,
    write_latest_marker,
    write_metrics_json,
)
from mindface.models.factory import build_model, is_sequence_model
from mindface.utils.config import resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    requested = str(device_name).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requested device=cuda, but this Python environment cannot use CUDA. "
                "Install a CUDA-enabled PyTorch build, then rerun training."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported train.device={device_name}. Use auto, cuda, or cpu.")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        pred = model(x)
        loss = loss_fn(pred, y)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def train_from_config(cfg: dict, logger) -> Path:
    started_at = time.perf_counter()
    train_cfg = cfg["train"]
    model_type = str(cfg["model"]["type"]).lower()
    model_cfg = dict(cfg["model"]["params"])
    sequence_mode = is_sequence_model(model_type)

    set_seed(int(train_cfg.get("seed", 42)))
    device = resolve_device(str(train_cfg.get("device", "auto")))

    dataset_dir = resolve_path(cfg["dataset"]["dir"])
    train_ds = MouthParameterDataset(dataset_dir, split="train", sequence_mode=sequence_mode)
    val_ds = MouthParameterDataset(dataset_dir, split="val", sequence_mode=sequence_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = build_model(model_type, model_cfg).to(device)
    parameter_count = count_trainable_parameters(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    checkpoint_path = resolve_path(cfg["output"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_root = cfg["output"].get("experiment_root", "outputs/experiments")
    experiment_dir = make_experiment_dir(experiment_root, f"train_{model_type}")
    write_config_snapshot(experiment_dir, cfg)

    if device.type == "cuda":
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("Training model_type=%s sequence_mode=%s device=%s", model_type, sequence_mode, device)
    logger.info("Train items=%d Val items=%d", len(train_ds), len(val_ds))
    logger.info("Trainable parameters=%d", parameter_count)
    logger.info("Experiment dir: %s", experiment_dir)

    best_val = float("inf")
    history: list[dict[str, float | int | bool]] = []
    epochs = int(train_cfg.get("epochs", 5))
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, loss_fn, device)
        logger.info("Epoch %03d/%03d train_loss=%.6f val_loss=%.6f", epoch, epochs, train_loss, val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_val_loss": float(best_val),
                "is_best": bool(is_best),
            }
        )

        if is_best:
            torch.save(
                {
                    "model_type": model_type,
                    "model_config": model_cfg,
                    "model_state": model.state_dict(),
                    "best_val_loss": best_val,
                    "feature_config": cfg.get("features", {}),
                    "train_config": train_cfg,
                },
                checkpoint_path,
            )
            logger.info("Saved checkpoint: %s", checkpoint_path)

    duration_sec = time.perf_counter() - started_at
    metrics = {
        "model_type": model_type,
        "sequence_mode": sequence_mode,
        "dataset_dir": str(dataset_dir),
        "checkpoint_path": str(checkpoint_path),
        "train_items": len(train_ds),
        "val_items": len(val_ds),
        "epochs": epochs,
        "best_val_loss": float(best_val),
        "parameter_count": parameter_count,
        "device": str(device),
        "duration_sec": float(duration_sec),
        "runtime": {
            **base_runtime_info(),
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_history_csv(experiment_dir, history)
    write_metrics_json(experiment_dir, metrics)
    write_latest_marker(experiment_root, experiment_dir)
    logger.info("Experiment metrics: %s", experiment_dir / "metrics.json")
    return checkpoint_path
