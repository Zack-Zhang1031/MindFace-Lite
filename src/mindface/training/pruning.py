from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from mindface.data.synthetic_dataset import SyntheticMouthDataset
from mindface.inference import load_torch_model, predict_from_features
from mindface.models.factory import is_sequence_model
from mindface.training.trainer import _run_epoch, resolve_device, set_seed
from mindface.utils.config import resolve_path


def prunable_modules(model: nn.Module) -> list[nn.Module]:
    return [module for module in model.modules() if isinstance(module, (nn.Linear, nn.Conv1d))]


def sparsity_report(model: nn.Module) -> dict[str, float]:
    total = 0
    zeros = 0
    for module in prunable_modules(model):
        data = module.weight.detach()
        total += data.numel()
        zeros += int(torch.sum(data == 0).item())
    return {
        "total_weight_params": float(total),
        "zero_weight_params": float(zeros),
        "sparsity": float(zeros / max(total, 1)),
    }


def apply_l1_unstructured_pruning(model: nn.Module, amount: float) -> dict[str, float]:
    if not 0.0 <= amount < 1.0:
        raise ValueError("Pruning amount must be in [0, 1).")
    for module in prunable_modules(model):
        prune.l1_unstructured(module, name="weight", amount=amount)
    return sparsity_report(model)


def remove_pruning_reparam(model: nn.Module) -> None:
    """Make pruning masks permanent after fine-tuning."""
    for module in prunable_modules(model):
        if hasattr(module, "weight_mask"):
            prune.remove(module, "weight")


def prune_and_finetune_from_config(cfg: dict, logger) -> Path:
    set_seed(int(cfg["train"].get("seed", 42)))
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    model, checkpoint = load_torch_model(cfg["checkpoint"]["input_path"], device=str(device))
    model_type = checkpoint["model_type"]
    sequence_mode = is_sequence_model(model_type)
    before = sparsity_report(model)
    after = apply_l1_unstructured_pruning(model, amount=float(cfg["pruning"]["amount"]))
    model.to(device)

    logger.info("Loaded checkpoint: %s", resolve_path(cfg["checkpoint"]["input_path"]))
    if device.type == "cuda":
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("Pruning amount=%.3f before=%s after=%s", float(cfg["pruning"]["amount"]), before, after)

    dataset_dir = resolve_path(cfg["dataset"]["dir"])
    train_ds = SyntheticMouthDataset(dataset_dir, split="train", sequence_mode=sequence_mode)
    val_ds = SyntheticMouthDataset(dataset_dir, split="val", sequence_mode=sequence_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"].get("lr", 1e-4)))
    best_val = float("inf")
    best_state = None
    epochs = int(cfg["train"].get("epochs", 1))
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, loss_fn, device)
        logger.info("Prune finetune epoch %03d/%03d train_loss=%.6f val_loss=%.6f", epoch, epochs, train_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    remove_pruning_reparam(model)
    final = sparsity_report(model)
    output_path = resolve_path(cfg["checkpoint"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "model_config": checkpoint["model_config"],
            "model_state": model.state_dict(),
            "best_val_loss": best_val,
            "feature_config": checkpoint.get("feature_config", {}),
            "train_config": cfg["train"],
            "pruning": {
                "method": "l1_unstructured",
                "amount": float(cfg["pruning"]["amount"]),
                "before": before,
                "after_prune": after,
                "after_finetune": final,
            },
        },
        output_path,
    )
    logger.info("Saved pruned checkpoint: %s", output_path)
    return output_path


def benchmark_checkpoints(
    checkpoint_paths: dict[str, str | Path],
    frames: int,
    feature_dim: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    features = rng.random((frames, feature_dim), dtype=np.float32)
    report = {"input_shape": [frames, feature_dim], "checkpoints": {}}
    for name, path in checkpoint_paths.items():
        model, checkpoint = load_torch_model(path, device="cpu")
        for _ in range(warmup):
            predict_from_features(model, checkpoint["model_type"], features, device="cpu")
        values = []
        for _ in range(repeat):
            start = perf_counter()
            predict_from_features(model, checkpoint["model_type"], features, device="cpu")
            values.append((perf_counter() - start) * 1000.0)
        values_sorted = sorted(values)
        mean = float(np.mean(values))
        report["checkpoints"][name] = {
            "path": str(resolve_path(path)),
            "sparsity": sparsity_report(model),
            "latency": {
                "mean_ms": mean,
                "median_ms": float(values_sorted[len(values_sorted) // 2]),
                "p95_ms": float(values_sorted[min(len(values_sorted) - 1, int(len(values_sorted) * 0.95))]),
                "fps": float(1000.0 / max(mean, 1e-9)),
            },
        }
    return report
