from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def speaker_disjoint_split(
    speakers: Iterable[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    unique = sorted({str(speaker) for speaker in speakers if str(speaker)})
    if not unique:
        raise ValueError("At least one non-empty speaker is required")
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must contain non-negative train/val/test values summing to 1.0")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    labels = ("train", "val", "test")
    count = len(unique)
    if count < 3:
        return {speaker: labels[index] for index, speaker in enumerate(unique)}

    raw_counts = [count * ratio for ratio in ratios]
    counts = [int(np.floor(value)) for value in raw_counts]
    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            counts[index] = 1
    while sum(counts) > count:
        candidates = [index for index, value in enumerate(counts) if value > 1]
        counts[max(candidates, key=lambda index: counts[index] - raw_counts[index])] -= 1
    while sum(counts) < count:
        index = max(range(3), key=lambda item: raw_counts[item] - counts[item])
        counts[index] += 1

    assignments: dict[str, str] = {}
    cursor = 0
    for label, split_count in zip(labels, counts):
        for speaker in unique[cursor : cursor + split_count]:
            assignments[speaker] = label
        cursor += split_count
    return assignments

