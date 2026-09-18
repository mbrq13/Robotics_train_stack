"""Adaptive filtering for noisy VR-derived joint targets."""

from __future__ import annotations

import math

import numpy as np


class OneEuroJointFilter:
    """Vectorized 1€ low-pass filter with an explicit sample clock.

    It is for human tracking targets only. Policy rollout actions never pass
    through this class because filtering would alter the trained action stream.
    """

    def __init__(
        self,
        *,
        min_cutoff_hz: float = 2.0,
        velocity_coefficient: float = 3.0,
        derivative_cutoff_hz: float = 5.0,
    ) -> None:
        values = (min_cutoff_hz, velocity_coefficient, derivative_cutoff_hz)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("filter parameters must be finite and non-negative")
        if min_cutoff_hz <= 0.0 or derivative_cutoff_hz <= 0.0:
            raise ValueError("filter cutoffs must be positive")
        self.min_cutoff_hz = min_cutoff_hz
        self.velocity_coefficient = velocity_coefficient
        self.derivative_cutoff_hz = derivative_cutoff_hz
        self._previous: np.ndarray | None = None
        self._derivative: np.ndarray | None = None
        self._timestamp_ns: int | None = None

    def reset(self, values: np.ndarray | None = None, *, timestamp_ns: int | None = None) -> None:
        self._previous = None if values is None else np.asarray(values, dtype=np.float64).copy()
        self._derivative = None
        self._timestamp_ns = timestamp_ns

    def apply(self, values: np.ndarray, *, timestamp_ns: int) -> np.ndarray:
        """Filter one finite vector; first use after reset is an identity step."""
        current = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(current).all() or timestamp_ns <= 0:
            raise ValueError("filter samples must be finite and have a positive timestamp")
        if self._previous is None or self._timestamp_ns is None:
            self.reset(current, timestamp_ns=timestamp_ns)
            return current.copy()
        if current.shape != self._previous.shape:
            raise ValueError("filter input dimension changed")
        dt_s = (timestamp_ns - self._timestamp_ns) / 1_000_000_000
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return self._previous.copy()
        derivative = (current - self._previous) / dt_s
        derivative_alpha = self._alpha(self.derivative_cutoff_hz, dt_s)
        previous_derivative = (
            np.zeros_like(current) if self._derivative is None else self._derivative
        )
        filtered_derivative = (
            derivative_alpha * derivative + (1.0 - derivative_alpha) * previous_derivative
        )
        cutoff = self.min_cutoff_hz + self.velocity_coefficient * np.abs(filtered_derivative)
        alpha = self._alpha(cutoff, dt_s)
        filtered = alpha * current + (1.0 - alpha) * self._previous
        self._previous = filtered
        self._derivative = filtered_derivative
        self._timestamp_ns = timestamp_ns
        return filtered.copy()

    @staticmethod
    def _alpha(cutoff_hz: float | np.ndarray, dt_s: float) -> float | np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt_s)
