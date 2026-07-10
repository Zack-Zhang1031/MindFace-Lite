from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mindface.utils.config import resolve_path


def write_rule_csv(
    output_path: str | Path,
    times_sec: np.ndarray,
    rms: np.ndarray,
    mouth_open: np.ndarray,
) -> None:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "time_sec", "rms", "mouth_open"])
        writer.writeheader()
        for idx, (time_sec, rms_value, open_value) in enumerate(zip(times_sec, rms, mouth_open)):
            writer.writerow(
                {
                    "frame_index": idx,
                    "time_sec": f"{float(time_sec):.6f}",
                    "rms": f"{float(rms_value):.8f}",
                    "mouth_open": f"{float(open_value):.6f}",
                }
            )


def write_params_csv(output_path: str | Path, times_sec: np.ndarray, params: np.ndarray) -> None:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = np.asarray(params, dtype=np.float32)
    if params.ndim == 1:
        params = params[:, None]
    names = ["mouth_open", "mouth_width", "lip_round"]
    fieldnames = ["frame_index", "time_sec"] + names[: params.shape[1]]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(params):
            item = {"frame_index": idx, "time_sec": f"{float(times_sec[idx]):.6f}"}
            for col_idx, value in enumerate(row):
                item[names[col_idx]] = f"{float(value):.6f}"
            writer.writerow(item)
