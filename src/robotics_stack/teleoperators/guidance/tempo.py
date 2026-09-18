"""Timestamped target playback for operator-controlled motion."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetBridgeSettings:
    """Cadence and freshness contract between an input device and the robot."""

    input_rate_hz: float = 72.0
    output_rate_hz: float = 100.0
    playback_delay_s: float | None = None
    source_timeout_s: float = 0.150
    history_size: int = 128

    def __post_init__(self) -> None:
        values = (self.input_rate_hz, self.output_rate_hz, self.source_timeout_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("target bridge rates and source timeout must be positive")
        if self.playback_delay_s is not None and (
            not math.isfinite(self.playback_delay_s) or self.playback_delay_s < 0
        ):
            raise ValueError("playback_delay_s must be finite and non-negative")
        if self.history_size < 2:
            raise ValueError("history_size must be at least two")

    @property
    def resolved_playback_delay_s(self) -> float:
        if self.playback_delay_s is not None:
            return self.playback_delay_s
        return 1.0 / self.input_rate_hz + 1.0 / self.output_rate_hz


@dataclass(frozen=True)
class TargetSample:
    timestamp_s: float
    values: np.ndarray


class TargetTimeline:
    """Thread-safe target history sampled at a deliberately delayed clock."""

    def __init__(self, settings: TargetBridgeSettings):
        self.settings = settings
        self._samples: deque[TargetSample] = deque(maxlen=settings.history_size)
        self._size: int | None = None
        self._lock = threading.Lock()

    def reset(self, values: tuple[float, ...], *, timestamp_s: float) -> None:
        sample = self._sample(values, timestamp_s)
        with self._lock:
            self._samples.clear()
            self._samples.append(sample)
            self._size = sample.values.size

    def push(self, values: tuple[float, ...], *, timestamp_s: float) -> None:
        sample = self._sample(values, timestamp_s)
        with self._lock:
            if self._size is not None and sample.values.size != self._size:
                raise ValueError("operator target dimension changed")
            if self._samples and sample.timestamp_s <= self._samples[-1].timestamp_s:
                raise ValueError("operator target timestamps must increase")
            self._samples.append(sample)
            self._size = sample.values.size

    def sample(self, *, now_s: float) -> tuple[tuple[float, ...], str] | None:
        if not math.isfinite(now_s):
            raise ValueError("sample time must be finite")
        playback_s = now_s - self.settings.resolved_playback_delay_s
        with self._lock:
            if not self._samples:
                return None
            while len(self._samples) >= 3 and self._samples[1].timestamp_s <= playback_s:
                self._samples.popleft()
            first = self._samples[0]
            if len(self._samples) == 1 or playback_s <= first.timestamp_s:
                return self._values(first.values), "hold"
            second = self._samples[1]
            if playback_s >= second.timestamp_s:
                return self._values(second.values), "hold"
            fraction = (playback_s - first.timestamp_s) / (second.timestamp_s - first.timestamp_s)
            values = first.values + fraction * (second.values - first.values)
            return self._values(values), "interpolate"

    def source_age_s(self, *, now_s: float) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            return max(0.0, now_s - self._samples[-1].timestamp_s)

    @staticmethod
    def _sample(values: tuple[float, ...], timestamp_s: float) -> TargetSample:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not array.size or not np.isfinite(array).all():
            raise ValueError("operator target must be a non-empty finite vector")
        if not math.isfinite(timestamp_s):
            raise ValueError("operator target timestamp must be finite")
        return TargetSample(float(timestamp_s), array.copy())

    @staticmethod
    def _values(values: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in values)


class FixedRateTargetBridge:
    """Delivers interpolated targets at a stable rate and holds on stale input."""

    def __init__(
        self,
        write: Callable[[tuple[float, ...]], None],
        hold: Callable[[], None],
        *,
        settings: TargetBridgeSettings,
        on_emit: Callable[[tuple[float, ...], int], None] | None = None,
    ):
        self.settings = settings
        self.timeline = TargetTimeline(settings)
        self._write = write
        self._hold = hold
        self._on_emit = on_emit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._stale = False

    def submit(
        self,
        values: tuple[float, ...],
        *,
        timestamp_s: float | None = None,
        reset: bool = False,
    ) -> None:
        """Accept one local operator target, timestamping receipt by default."""
        self.raise_if_failed()
        if self._stale:
            raise RuntimeError("target bridge is stale; create a new operator epoch")
        if timestamp_s is None:
            timestamp_s = time.perf_counter()
        if reset:
            self.timeline.reset(values, timestamp_s=timestamp_s)
        else:
            self.timeline.push(values, timestamp_s=timestamp_s)

    def start(self) -> None:
        self.raise_if_failed()
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="operator-target-bridge",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.raise_if_failed()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stale(self) -> bool:
        return self._stale

    def tick(self, *, now_s: float) -> bool:
        """Execute one output tick; exposed for deterministic verification."""
        if self._stale:
            return False
        age_s = self.timeline.source_age_s(now_s=now_s)
        if age_s is None:
            return True
        if age_s > self.settings.source_timeout_s:
            self._stale = True
            self._hold()
            return False
        sample = self.timeline.sample(now_s=now_s)
        if sample is not None:
            values, _mode = sample
            self._write(values)
            if self._on_emit is not None:
                self._on_emit(values, time.monotonic_ns())
        return True

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("operator target bridge failed") from self._failure

    def _run(self) -> None:
        period_s = 1.0 / self.settings.output_rate_hz
        deadline_s = time.perf_counter()
        try:
            while not self._stop.is_set() and self.tick(now_s=time.perf_counter()):
                deadline_s += period_s
                now_s = time.perf_counter()
                if deadline_s <= now_s:
                    missed = int((now_s - deadline_s) // period_s) + 1
                    deadline_s += missed * period_s
                self._stop.wait(max(0.0, deadline_s - now_s))
        except BaseException as exc:
            self._failure = exc
            self._hold()
