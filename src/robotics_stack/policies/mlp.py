"""Small native policy for baseline training and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robotics_stack.contracts import CheckpointManifest


class StateMlpPolicy:
    kind = "state_mlp"

    def __init__(self, model, state_mean, state_std, action_mean, action_std):
        self.model = model
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std

    def predict(self, state: tuple[float, ...]) -> tuple[float, ...]:
        import torch

        values = np.asarray(state, dtype=np.float32)
        normalized = (values - self.state_mean) / self.state_std
        with torch.inference_mode():
            output = self.model(torch.from_numpy(normalized).unsqueeze(0)).squeeze(0).cpu().numpy()
        action = output * self.action_std + self.action_mean
        return tuple(float(value) for value in action)

    @classmethod
    def load(cls, directory: str | Path) -> tuple[StateMlpPolicy, CheckpointManifest]:
        import torch

        root = Path(directory)
        payload = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
        manifest = CheckpointManifest.from_dict(payload["manifest"])
        if manifest.policy_kind != cls.kind:
            raise ValueError(f"expected {cls.kind}, got {manifest.policy_kind}")
        model = build_model(manifest.state_dim, manifest.action_dim, payload["hidden_dim"])
        model.load_state_dict(payload["model"])
        model.eval()
        return (
            cls(
                model,
                np.asarray(payload["state_mean"], dtype=np.float32),
                np.asarray(payload["state_std"], dtype=np.float32),
                np.asarray(payload["action_mean"], dtype=np.float32),
                np.asarray(payload["action_std"], dtype=np.float32),
            ),
            manifest,
        )


def build_model(state_dim: int, action_dim: int, hidden_dim: int = 256):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(state_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, action_dim),
    )
