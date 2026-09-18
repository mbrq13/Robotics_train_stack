"""Stateful, opt-in smoothing for actions emitted by synchronous inference."""

from __future__ import annotations

import math
from collections.abc import Sequence


class ExponentialActionSmoother:
    """Apply an exponential moving average and reset at a control boundary.

    ``alpha=1`` is the identity transform. Keeping that as the default makes
    the inference path exactly match the checkpoint output unless an operator
    explicitly chooses otherwise.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("action_smoothing_alpha must be in (0, 1]")
        self.alpha = alpha
        self._previous: tuple[float, ...] | None = None

    def reset(self) -> None:
        self._previous = None

    def apply(self, action: Sequence[float]) -> tuple[float, ...]:
        current = tuple(float(value) for value in action)
        if not all(math.isfinite(value) for value in current):
            raise ValueError("cannot smooth non-finite action values")
        if self._previous is None:
            self._previous = current
            return current
        if len(current) != len(self._previous):
            raise ValueError("action dimension changed while smoothing")
        smoothed = tuple(
            previous + self.alpha * (value - previous)
            for value, previous in zip(current, self._previous, strict=True)
        )
        self._previous = smoothed
        return smoothed
