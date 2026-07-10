from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class LatencyStats:
    """Small helper for realtime and benchmark reports."""

    values_ms: list[float] = field(default_factory=list)

    def add_since(self, start_time: float) -> None:
        self.values_ms.append((perf_counter() - start_time) * 1000.0)

    def summary(self) -> dict[str, float]:
        if not self.values_ms:
            return {"count": 0.0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        values = sorted(self.values_ms)
        mid = len(values) // 2
        p95_idx = min(len(values) - 1, int(len(values) * 0.95))
        return {
            "count": float(len(values)),
            "mean_ms": float(sum(values) / len(values)),
            "median_ms": float(values[mid]),
            "p95_ms": float(values[p95_idx]),
            "min_ms": float(values[0]),
            "max_ms": float(values[-1]),
        }
