"""Small, dependency-light latency summaries for policy profiling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean

import numpy as np


@dataclass(frozen=True)
class LatencySummary:
    samples: int
    minimum_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float
    delay_steps_p95: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize_latency(samples_s: list[float], control_hz: float) -> LatencySummary:
    """Summarize finite durations and convert the p95 to control steps."""
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if not samples_s or not all(math.isfinite(sample) and sample >= 0 for sample in samples_s):
        raise ValueError("latency samples must be finite, non-negative and non-empty")
    values_ms = np.asarray(samples_s, dtype=np.float64) * 1_000.0
    p95_ms = float(np.percentile(values_ms, 95))
    return LatencySummary(
        samples=len(samples_s),
        minimum_ms=float(values_ms.min()),
        mean_ms=float(fmean(values_ms)),
        p50_ms=float(np.percentile(values_ms, 50)),
        p95_ms=p95_ms,
        p99_ms=float(np.percentile(values_ms, 99)),
        maximum_ms=float(values_ms.max()),
        delay_steps_p95=max(1, math.ceil((p95_ms / 1_000.0) * control_hz)),
    )
