"""Timestamp-aware capture primitives for trainable robot episodes.

The control loop only submits an action that the station accepted. Sensor reads
and disk work happen elsewhere, but an overdue capture is rejected rather than
being written as a plausible-looking, misaligned row.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robotics_stack.robots.cameras import CameraFrame


class CaptureError(RuntimeError):
    """A recording row cannot satisfy its temporal or schema contract."""


@dataclass(frozen=True)
class CaptureSettings:
    """Bounded timing contract for a station-local dataset capture."""

    fps: float = 30.0
    max_capture_delay_ms: float = 80.0
    max_camera_age_ms: float = 250.0
    max_camera_skew_ms: float = 60.0

    def __post_init__(self) -> None:
        values = (
            self.fps,
            self.max_capture_delay_ms,
            self.max_camera_age_ms,
            self.max_camera_skew_ms,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("capture settings must be finite and positive")

    @property
    def period_ns(self) -> int:
        return round(1_000_000_000 / self.fps)


@dataclass(frozen=True)
class CaptureRequest:
    """An action accepted by the station and eligible for data capture."""

    action: tuple[float, ...]
    emitted_monotonic_ns: int
    source: int
    control_generation: int

    def __post_init__(self) -> None:
        values = np.asarray(self.action, dtype=np.float64).reshape(-1)
        if not values.size or not np.isfinite(values).all():
            raise CaptureError("recorded action must be a non-empty finite vector")
        if self.emitted_monotonic_ns <= 0:
            raise CaptureError("recorded action requires a positive monotonic timestamp")
        if self.source not in {0, 1}:
            raise CaptureError("recording source must be policy (0) or operator (1)")
        if self.control_generation < 0:
            raise CaptureError("control generation must not be negative")


@dataclass(frozen=True)
class CapturedFrame:
    """One temporally audited row ready for a dataset writer."""

    request: CaptureRequest
    state: tuple[float, ...]
    state_monotonic_ns: int
    cameras: dict[str, CameraFrame]


class CaptureCoordinator:
    """Samples state and encoded camera frames without inventing alignment.

    `read_state` and `read_cameras` must be station-local callables sharing the
    station monotonic clock. A source that is too old or too far apart from the
    others raises `CaptureError`; silently substituting a later image would
    corrupt supervised data.
    """

    def __init__(
        self,
        *,
        settings: CaptureSettings,
        action_size: int,
        camera_names: tuple[str, ...],
        read_state: Callable[[], tuple[float, ...]],
        read_cameras: Callable[[], dict[str, CameraFrame]],
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.settings = settings
        self.action_size = action_size
        self.camera_names = camera_names
        self._read_state = read_state
        self._read_cameras = read_cameras
        self._clock_ns = clock_ns
        self._last_capture_ns: int | None = None
        self._rate_lock = threading.Lock()

    def due(self, emitted_monotonic_ns: int) -> bool:
        """Return whether an accepted command starts the next dataset period."""
        with self._rate_lock:
            if self._last_capture_ns is not None and (
                emitted_monotonic_ns - self._last_capture_ns < self.settings.period_ns
            ):
                return False
            # Reserve the interval before a background worker starts. This
            # prevents a burst of bridge ticks from defeating the data FPS.
            self._last_capture_ns = emitted_monotonic_ns
            return True

    def capture(self, request: CaptureRequest) -> CapturedFrame:
        """Read one state/camera bundle and verify its temporal bounds."""
        started_ns = self._clock_ns()
        if started_ns - request.emitted_monotonic_ns > self._ms_to_ns(
            self.settings.max_capture_delay_ms
        ):
            raise CaptureError("capture worker started after its action-alignment deadline")
        state = tuple(float(value) for value in self._read_state())
        state_ns = self._clock_ns()
        if len(state) != self.action_size or not np.isfinite(state).all():
            raise CaptureError("recorded state does not match the action schema")
        if state_ns - request.emitted_monotonic_ns > self._ms_to_ns(
            self.settings.max_capture_delay_ms
        ):
            raise CaptureError("state feedback arrived after its action-alignment deadline")
        cameras = self._read_cameras()
        if tuple(sorted(cameras)) != tuple(sorted(self.camera_names)):
            raise CaptureError("recorded camera set differs from the station schema")
        if cameras:
            timestamps = [frame.monotonic_ns for frame in cameras.values()]
            oldest_ns = min(timestamps)
            newest_ns = max(timestamps)
            if state_ns - oldest_ns > self._ms_to_ns(self.settings.max_camera_age_ms):
                raise CaptureError("a camera frame is too old for the recorded state")
            if newest_ns - oldest_ns > self._ms_to_ns(self.settings.max_camera_skew_ms):
                raise CaptureError("camera frames exceed the configured synchronization skew")
            if any(not frame.data for frame in cameras.values()):
                raise CaptureError("recorded camera frame is empty")
        return CapturedFrame(request, state, state_ns, cameras)

    @staticmethod
    def _ms_to_ns(value: float) -> int:
        return round(value * 1_000_000)
