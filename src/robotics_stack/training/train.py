"""Native behavior-cloning trainer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robotics_stack.contracts import CheckpointManifest, RobotSchema
from robotics_stack.policies.mlp import StateMlpPolicy, build_model


@dataclass(frozen=True)
class TrainResult:
    output: Path
    loss: float
    samples: int


def train_state_mlp(
    data_path: str | Path,
    output: str | Path,
    schema: RobotSchema,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    hidden_dim: int = 256,
    seed: int = 7,
) -> TrainResult:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.load(data_path)
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("states and actions must be rank-2 arrays")
    if states.shape[0] != actions.shape[0] or states.shape[0] == 0:
        raise ValueError("states and actions must contain the same non-zero sample count")
    if states.shape[1] != len(schema.state_names) or actions.shape[1] != schema.action_size:
        raise ValueError("dataset dimensions do not match the selected robot schema")
    for action in actions:
        schema.validate_action(action)

    state_mean, state_std = states.mean(0), np.maximum(states.std(0), 1e-6)
    action_mean, action_std = actions.mean(0), np.maximum(actions.std(0), 1e-6)
    states = (states - state_mean) / state_std
    targets = (actions - action_mean) / action_std
    model = build_model(states.shape[1], actions.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    inputs = torch.from_numpy(states)
    labels = torch.from_numpy(targets)
    last_loss = float("inf")

    for _ in range(epochs):
        order = torch.randperm(inputs.shape[0])
        for start in range(0, inputs.shape[0], batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs[indices]), labels[indices])
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = CheckpointManifest(
        policy_kind=StateMlpPolicy.kind,
        schema=schema,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        metrics={"train_mse": last_loss},
    )
    manifest.validate()
    torch.save(
        {
            "model": model.state_dict(),
            "hidden_dim": hidden_dim,
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "manifest": manifest.to_dict(),
        },
        root / "model.pt",
    )
    (root / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return TrainResult(root, last_loss, int(states.shape[0]))
