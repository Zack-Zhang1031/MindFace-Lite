from __future__ import annotations

from collections import defaultdict

from mindface.data.splitting import speaker_disjoint_split


def test_speaker_split_is_deterministic_and_disjoint() -> None:
    speakers = [f"s{index}" for index in range(10) for _ in range(3)]

    first = speaker_disjoint_split(speakers, ratios=(0.8, 0.1, 0.1), seed=42)
    second = speaker_disjoint_split(reversed(speakers), ratios=(0.8, 0.1, 0.1), seed=42)

    assert first == second
    groups: dict[str, set[str]] = defaultdict(set)
    for speaker, split in first.items():
        groups[split].add(speaker)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert {split for split in first.values()} == {"train", "val", "test"}


def test_speaker_split_handles_one_and_two_speakers() -> None:
    assert speaker_disjoint_split(["s1"], (0.8, 0.1, 0.1), seed=1) == {"s1": "train"}
    two = speaker_disjoint_split(["s1", "s2"], (0.8, 0.1, 0.1), seed=1)

    assert sorted(two.values()) == ["train", "val"]

