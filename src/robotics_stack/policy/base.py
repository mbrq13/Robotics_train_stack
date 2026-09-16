"""Small stable contract for policies implemented in this repository."""

from __future__ import annotations

from typing import Protocol

from robotics_stack.contracts import CheckpointManifest


class Policy(Protocol):
    manifest: CheckpointManifest

    def predict(self, state: tuple[float, ...]) -> tuple[float, ...]: ...
