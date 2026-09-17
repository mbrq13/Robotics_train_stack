"""Versioned contracts shared by training and runtime components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


class ContractError(ValueError):
    """Raised when a policy artifact or network message violates a contract."""


@dataclass(frozen=True)
class JointLimit:
    minimum: float
    maximum: float
    max_speed: float
    max_acceleration: float

    def validate(self, value: float) -> bool:
        return np.isfinite(value) and self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class RobotSchema:
    """The explicit mechanical and perception contract for one robot setup."""

    name: str
    action_names: tuple[str, ...]
    state_names: tuple[str, ...]
    joint_limits: tuple[JointLimit, ...]
    camera_names: tuple[str, ...] = ()
    control_hz: float = 30.0
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.action_names or len(self.action_names) != len(self.joint_limits):
            raise ContractError("action_names and joint_limits must have the same non-zero length")
        if len(self.state_names) != len(self.action_names):
            raise ContractError("state_names and action_names must have the same length")
        if self.control_hz <= 0:
            raise ContractError("control_hz must be positive")

    @property
    def action_size(self) -> int:
        return len(self.action_names)

    def validate_action(self, values: list[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != self.action_size:
            raise ContractError(f"expected {self.action_size} action values, got {array.size}")
        for name, limit, value in zip(self.action_names, self.joint_limits, array, strict=True):
            if not limit.validate(float(value)):
                raise ContractError(f"{name}={value} is outside [{limit.minimum}, {limit.maximum}]")
        return array

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RobotSchema:
        return cls(
            name=value["name"],
            action_names=tuple(value["action_names"]),
            state_names=tuple(value["state_names"]),
            joint_limits=tuple(JointLimit(**limit) for limit in value["joint_limits"]),
            camera_names=tuple(value.get("camera_names", [])),
            control_hz=float(value.get("control_hz", 30.0)),
            version=int(value.get("version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class Observation:
    observation_id: int
    station_monotonic_ns: int
    state: tuple[float, ...]
    images: dict[str, bytes] = field(default_factory=dict)
    control_generation: int = 0


@dataclass(frozen=True)
class PolicyAction:
    session_id: str
    sequence_id: int
    observation_id: int
    station_monotonic_ns: int
    schema_version: int
    values: tuple[float, ...]
    scheduled: bool = False
    control_generation: int = 0

    def __post_init__(self) -> None:
        if self.control_generation < 0:
            raise ContractError("control_generation must not be negative")


@dataclass(frozen=True)
class CheckpointManifest:
    """Metadata required to make a checkpoint deployable, not merely loadable."""

    policy_kind: str
    schema: RobotSchema
    state_dim: int
    action_dim: int
    artifact_version: int = 1
    framework: str = "torch"
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.state_dim != len(self.schema.state_names):
            raise ContractError("manifest state dimension differs from schema")
        if self.action_dim != self.schema.action_size:
            raise ContractError("manifest action dimension differs from schema")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = self.schema.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CheckpointManifest:
        manifest = cls(
            policy_kind=value["policy_kind"],
            schema=RobotSchema.from_dict(value["schema"]),
            state_dim=int(value["state_dim"]),
            action_dim=int(value["action_dim"]),
            artifact_version=int(value.get("artifact_version", 1)),
            framework=value.get("framework", "torch"),
            metrics={key: float(metric) for key, metric in value.get("metrics", {}).items()},
        )
        manifest.validate()
        return manifest
