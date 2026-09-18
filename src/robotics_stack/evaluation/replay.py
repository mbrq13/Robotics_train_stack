"""Offline artifact validation against recorded state/action data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robotics_stack.policies.mlp import StateMlpPolicy


@dataclass(frozen=True)
class ReplayReport:
    samples: int
    action_mse: float
    rejected_outputs: int


def evaluate_state_mlp(checkpoint: str | Path, data_path: str | Path) -> ReplayReport:
    policy, manifest = StateMlpPolicy.load(checkpoint)
    data = np.load(data_path)
    states = np.asarray(data["states"], dtype=np.float64)
    expected = np.asarray(data["actions"], dtype=np.float64)
    if states.ndim != 2 or expected.ndim != 2:
        raise ValueError("states and actions must be rank-2 arrays")
    if states.shape[1] != manifest.state_dim or expected.shape[1] != manifest.action_dim:
        raise ValueError("replay data does not match checkpoint manifest")
    predictions = np.asarray([policy.predict(tuple(row)) for row in states], dtype=np.float64)
    rejected = 0
    for action in predictions:
        try:
            manifest.schema.validate_action(action)
        except ValueError:
            rejected += 1
    return ReplayReport(
        samples=int(states.shape[0]),
        action_mse=float(np.mean((predictions - expected) ** 2)),
        rejected_outputs=rejected,
    )
