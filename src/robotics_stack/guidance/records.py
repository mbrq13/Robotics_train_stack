"""Portable provenance for policy and operator demonstrations."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotics_stack.guidance.authority import ControlSource, GuidancePhase


@dataclass(frozen=True)
class GuidedSample:
    """One trainable state/action pair with its controller provenance."""

    observation_id: int
    station_monotonic_ns: int
    state: tuple[float, ...]
    action: tuple[float, ...]
    phase: GuidancePhase
    control_generation: int
    images: dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observation_id < 0 or self.control_generation < 0:
            raise ValueError("observation_id and control_generation must not be negative")
        if not self.state or not self.action:
            raise ValueError("a guided sample requires non-empty state and action vectors")
        if not np.isfinite(self.state).all() or not np.isfinite(self.action).all():
            raise ValueError("a guided sample contains a non-finite value")
        if self.phase not in {
            GuidancePhase.POLICY_RUN,
            GuidancePhase.RECOVERY,
            GuidancePhase.CORRECTION,
        }:
            raise ValueError("only motion-producing phases can be written as guided samples")
        valid_images = all(
            isinstance(name, str) and isinstance(image, bytes)
            for name, image in self.images.items()
        )
        if not valid_images:
            raise ValueError("guided images must map camera names to encoded bytes")

    @property
    def source(self) -> ControlSource:
        if self.phase is GuidancePhase.POLICY_RUN:
            return ControlSource.POLICY
        return ControlSource.OPERATOR


@dataclass
class GuidedEpisode:
    """Ordered, exportable samples from one guided collection pass."""

    task: str
    schema_version: int
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _samples: list[GuidedSample] = field(default_factory=list, init=False, repr=False)

    @property
    def samples(self) -> tuple[GuidedSample, ...]:
        return tuple(self._samples)

    def append(self, sample: GuidedSample) -> None:
        if self._samples:
            previous = self._samples[-1]
            if sample.observation_id <= previous.observation_id:
                raise ValueError("guided samples must have increasing observation ids")
            if sample.control_generation < previous.control_generation:
                raise ValueError("control_generation cannot move backwards within an episode")
            dimensions_changed = (
                len(sample.state) != len(previous.state)
                or len(sample.action) != len(previous.action)
            )
            if dimensions_changed:
                raise ValueError("all guided samples in an episode must use the same dimensions")
        self._samples.append(sample)

    def export_npz(self, path: str | Path) -> Path:
        """Write trainable arrays plus intervention provenance to an NPZ artifact."""
        if not self._samples:
            raise ValueError("cannot export an empty guided episode")
        output = Path(path)
        if output.suffix != ".npz":
            output = output.with_suffix(".npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "states": np.asarray([sample.state for sample in self._samples], dtype=np.float32),
            "actions": np.asarray([sample.action for sample in self._samples], dtype=np.float32),
            "operator_mask": np.asarray(
                [sample.source is ControlSource.OPERATOR for sample in self._samples],
                dtype=np.bool_,
            ),
            "control_generation": np.asarray(
                [sample.control_generation for sample in self._samples], dtype=np.int64
            ),
            "phase": np.asarray([sample.phase.value for sample in self._samples]),
        }
        camera_sets = {tuple(sorted(sample.images)) for sample in self._samples}
        if len(camera_sets) != 1:
            raise ValueError("all guided samples must contain the same camera set")
        for name in next(iter(camera_sets), ()):
            images = [sample.images[name] for sample in self._samples]
            offsets = np.cumsum([0, *(len(image) for image in images)], dtype=np.int64)
            payload[f"images_{name}_bytes"] = np.frombuffer(b"".join(images), dtype=np.uint8)
            payload[f"images_{name}_offsets"] = offsets
        np.savez_compressed(
            output,
            **payload,
        )
        output.with_suffix(".json").write_text(
            json.dumps(self.summary(), indent=2), encoding="utf-8"
        )
        return output

    def summary(self) -> dict[str, object]:
        operator_samples = sum(sample.source is ControlSource.OPERATOR for sample in self._samples)
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "schema_version": self.schema_version,
            "samples": len(self._samples),
            "operator_samples": operator_samples,
            "policy_samples": len(self._samples) - operator_samples,
        }
