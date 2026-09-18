"""Kinematic replay for validating policy-to-operator handoffs before hardware.

This is deliberately a visual and control-flow simulator, not a physics claim.
It replays recorded camera frames, feeds the current simulated joint state to a
policy, and records each policy or operator command with its provenance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from robotics_stack.contracts import ContractError, Observation, RobotSchema
from robotics_stack.guidance.authority import ControlAuthority, ControlSource, GuidancePhase
from robotics_stack.guidance.records import GuidedEpisode, GuidedSample


class PolicyPredictor(Protocol):
    def predict(self, observation: Observation) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ReplayFrame:
    """One replayed camera bundle. Joint state comes from the simulated robot."""

    images: dict[str, bytes]
    source_index: int


class DatasetReplay:
    """Finite camera replay loaded from an NPZ archive.

    The archive must contain ``images_<camera>`` arrays shaped ``[N,H,W,3]``.
    It may also contain ``states`` for provenance, but the simulator deliberately
    supplies the policy with its own current kinematic state instead.
    """

    def __init__(self, frames: list[ReplayFrame]):
        if not frames:
            raise ValueError("dataset replay requires at least one frame")
        camera_sets = {tuple(sorted(frame.images)) for frame in frames}
        if len(camera_sets) != 1:
            raise ValueError("all replay frames must contain the same cameras")
        self._frames = frames
        self._index = 0

    @classmethod
    def from_npz(cls, path: str | Path, *, camera_names: tuple[str, ...]) -> DatasetReplay:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("install the sim extra to encode dataset camera frames") from exc
        from io import BytesIO

        archive = np.load(Path(path), allow_pickle=False)
        encoded: dict[str, list[bytes]] = {}
        frame_count: int | None = None
        for name in camera_names:
            key = f"images_{name}"
            if key not in archive:
                raise ContractError(f"replay archive is missing {key}")
            images = np.asarray(archive[key])
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ContractError(f"{key} must have shape [N,H,W,3]")
            if frame_count is None:
                frame_count = int(images.shape[0])
            elif images.shape[0] != frame_count:
                raise ContractError("all replay cameras must have equal frame counts")
            output: list[bytes] = []
            for image in images:
                buffer = BytesIO()
                Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="JPEG")
                output.append(buffer.getvalue())
            encoded[name] = output
        assert frame_count is not None
        return cls(
            [
                ReplayFrame({name: encoded[name][index] for name in camera_names}, index)
                for index in range(frame_count)
            ]
        )

    @property
    def index(self) -> int:
        return self._index

    def next(self) -> ReplayFrame:
        frame = self._frames[self._index]
        self._index = (self._index + 1) % len(self._frames)
        return frame


class GuidedSimulation:
    """Explicit policy/operator handoff on a bounded kinematic robot state."""

    def __init__(
        self,
        schema: RobotSchema,
        replay: DatasetReplay,
        policy: PolicyPredictor,
        *,
        task: str = "",
    ) -> None:
        self.schema = schema
        self.replay = replay
        self.policy = policy
        self.authority = ControlAuthority()
        self.episode = GuidedEpisode(task=task, schema_version=schema.version)
        self._state = tuple(0.0 for _ in schema.state_names)
        self._observation_id = 0

    @property
    def state(self) -> tuple[float, ...]:
        return self._state

    @property
    def phase(self) -> GuidancePhase:
        return self.authority.snapshot.phase

    def start_policy(self) -> None:
        self.authority.activate_policy()

    def pause(self) -> None:
        self.authority.hold()

    def take_control(self, *, correction: bool = False) -> None:
        if self.phase is not GuidancePhase.HOLDING:
            raise RuntimeError("pause the policy before taking operator control")
        if correction:
            self.authority.begin_correction()
        else:
            self.authority.begin_recovery()

    def resume_policy(self) -> None:
        self.authority.finish_operator_control()
        self.authority.activate_policy()

    def step_policy(self) -> GuidedSample:
        if self.authority.snapshot.source is not ControlSource.POLICY:
            raise RuntimeError("policy is not the active control source")
        observation = self._observation()
        action = self.policy.predict(observation)
        return self._apply(action, observation)

    def step_operator(self, values: tuple[float, ...]) -> GuidedSample:
        if self.authority.snapshot.source is not ControlSource.OPERATOR:
            raise RuntimeError("operator control is not active")
        observation = self._observation()
        return self._apply(values, observation)

    def _observation(self) -> Observation:
        self._observation_id += 1
        frame = self.replay.next()
        return Observation(
            observation_id=self._observation_id,
            station_monotonic_ns=time.monotonic_ns(),
            state=self._state,
            images=frame.images,
            control_generation=self.authority.snapshot.generation,
        )

    def _apply(self, values: tuple[float, ...], observation: Observation) -> GuidedSample:
        action = tuple(float(value) for value in self.schema.validate_action(values))
        self._state = action
        snapshot = self.authority.snapshot
        sample = GuidedSample(
            observation_id=observation.observation_id,
            station_monotonic_ns=observation.station_monotonic_ns,
            state=observation.state,
            action=action,
            phase=snapshot.phase,
            control_generation=snapshot.generation,
            images=observation.images,
        )
        self.episode.append(sample)
        return sample
