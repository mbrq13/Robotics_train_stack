"""Framework-neutral real-time chunk scheduling primitives.

RTC lets a chunking policy infer while a separate fixed-rate actor continues
to consume already planned actions.  This module deliberately has no Torch,
websocket or hardware dependency, which makes its timing and merge semantics
testable without a robot.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RtcSettings:
    """The deployment side of the RTC contract stored with a Pi0.5 model."""

    control_hz: float
    training_max_delay: int
    execution_horizon: int
    refill_threshold: int

    def __post_init__(self) -> None:
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.training_max_delay <= 0:
            raise ValueError("trained RTC requires training_max_delay > 0")
        if self.execution_horizon < self.training_max_delay:
            raise ValueError("execution_horizon must be at least training_max_delay")
        if self.refill_threshold < self.training_max_delay:
            raise ValueError("refill_threshold must be at least training_max_delay")

    def validate_chunk_size(self, chunk_size: int) -> None:
        if chunk_size and self.execution_horizon > chunk_size - self.training_max_delay:
            raise ValueError(
                "execution_horizon must be <= chunk_size - training_max_delay "
                "for trained RTC"
            )

    @property
    def action_period_s(self) -> float:
        return 1.0 / self.control_hz


@dataclass(frozen=True)
class RtcChunk:
    """Raw model actions and their postprocessed robot-space counterparts."""

    raw: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        if self.raw.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("RTC chunks must have shape [time, action]")
        if self.raw.shape != self.actions.shape or not len(self.raw):
            raise ValueError("RTC raw and output chunks must be non-empty and equal-shaped")
        if not np.isfinite(self.raw).all() or not np.isfinite(self.actions).all():
            raise ValueError("RTC chunk contains non-finite values")


@dataclass(frozen=True)
class QueuedAction:
    raw: np.ndarray
    action: np.ndarray
    observation_id: int
    station_monotonic_ns: int


class RtcActionQueue:
    """A bounded, deterministic action queue with explicit chunk merging.

    ``inference_delay_steps`` actions already in flight are preserved.  The
    corresponding prefix of the newly predicted chunk is skipped because the
    Pi0.5 RTC objective conditions it on the preserved raw actions.
    """

    def __init__(self, settings: RtcSettings, action_dim: int):
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.settings = settings
        self.action_dim = action_dim
        self._items: deque[QueuedAction] = deque()
        self.underruns = 0

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def raw_left_over(self) -> np.ndarray | None:
        if not self._items:
            return None
        return np.stack([item.raw for item in self._items]).astype(np.float32, copy=False)

    def pop(self) -> QueuedAction | None:
        if not self._items:
            self.underruns += 1
            return None
        return self._items.popleft()

    def merge(
        self,
        chunk: RtcChunk,
        *,
        inference_delay_steps: int,
        observation_id: int,
        station_monotonic_ns: int,
    ) -> int:
        if chunk.actions.shape[1] != self.action_dim:
            raise ValueError("RTC chunk action dimension differs from station schema")
        delay = max(0, int(inference_delay_steps))
        if delay >= len(chunk.actions):
            raise ValueError("inference delay consumed the complete RTC action chunk")

        # Do not rewrite actions that the actor can consume before this result
        # becomes valid.  A stale queue is safer than a temporal discontinuity.
        preserved = list(self._items)[:delay]
        end = min(delay + self.settings.execution_horizon, len(chunk.actions))
        planned = [
            QueuedAction(
                raw=chunk.raw[index].astype(np.float32, copy=True),
                action=chunk.actions[index].astype(np.float32, copy=True),
                observation_id=observation_id,
                station_monotonic_ns=station_monotonic_ns,
            )
            for index in range(delay, end)
        ]
        self._items = deque([*preserved, *planned])
        return len(planned)


class LatencyTracker:
    """Small rolling latency estimator used to condition the next chunk."""

    def __init__(self, capacity: int = 128):
        self._samples: deque[float] = deque(maxlen=capacity)

    def add(self, latency_s: float) -> None:
        if latency_s >= 0 and math.isfinite(latency_s):
            self._samples.append(latency_s)

    def delay_steps(self, control_hz: float) -> int:
        if not self._samples:
            return 0
        return math.ceil(max(self._samples) * control_hz)

    def p95_ms(self) -> float:
        if not self._samples:
            return 0.0
        return float(np.percentile(np.asarray(self._samples), 95) * 1_000)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "samples": len(self._samples),
            "p95_ms": self.p95_ms(),
            "max_ms": (max(self._samples) * 1_000) if self._samples else 0.0,
            "updated_monotonic_ns": time.monotonic_ns(),
        }
