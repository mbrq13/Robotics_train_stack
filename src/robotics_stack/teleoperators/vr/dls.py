"""Incremental damped least-squares inverse kinematics for VR targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from robotics_stack.teleoperators.vr.transforms import pose_matrix, rotation_vector


class ArmKinematics(Protocol):
    """Minimal robot-specific surface required by the generic solver."""

    def forward(self, joints: np.ndarray) -> np.ndarray: ...

    def jacobian(self, joints: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DlsSettings:
    """Numerical and physical bounds for one incremental IK step."""

    base_damping: float = 0.05
    singularity_damping: float = 0.15
    singular_value_threshold: float = 0.05
    rest_gain: float = 0.02
    orientation_weight: float = 0.1
    orientation_hold_rad: float = 2.2
    max_step_multiplier: float = 2.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("DLS settings must be finite and non-negative")
        if self.base_damping <= 0.0 or self.singular_value_threshold <= 0.0:
            raise ValueError("DLS damping and singular-value threshold must be positive")
        if self.max_step_multiplier < 1.0:
            raise ValueError("DLS max_step_multiplier must be at least one")


class DampedLeastSquares:
    """One warm-started IK update with singularity and velocity bounds."""

    def __init__(
        self,
        kinematics: ArmKinematics,
        *,
        rest_joints: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        max_speed_rad_s: np.ndarray,
        nominal_rate_hz: float = 72.0,
        settings: DlsSettings | None = None,
    ) -> None:
        if not nominal_rate_hz > 0.0:
            raise ValueError("DLS nominal rate must be positive")
        self.kinematics = kinematics
        self.rest = np.asarray(rest_joints, dtype=np.float64).reshape(-1)
        self.lower = np.asarray(lower, dtype=np.float64).reshape(self.rest.size)
        self.upper = np.asarray(upper, dtype=np.float64).reshape(self.rest.size)
        self.speed = np.asarray(max_speed_rad_s, dtype=np.float64).reshape(self.rest.size)
        finite_joints = np.isfinite(self.rest).all() and np.isfinite(self.lower).all()
        finite_joints = finite_joints and np.isfinite(self.upper).all()
        if not finite_joints:
            raise ValueError("DLS joint values must be finite")
        invalid_speed = np.any(self.speed <= 0.0) or not np.isfinite(self.speed).all()
        if np.any(self.lower >= self.upper) or invalid_speed:
            raise ValueError("DLS joint limits and speed caps are invalid")
        self.settings = settings or DlsSettings()
        self._nominal_dt_s = 1.0 / nominal_rate_hz

    def step(self, current: np.ndarray, target_pose: np.ndarray, *, dt_s: float) -> np.ndarray:
        """Return the next bounded joint target; never extrapolates a stale step."""
        q = np.asarray(current, dtype=np.float64).reshape(self.rest.size)
        target = pose_matrix(target_pose)
        current_pose = self.kinematics.forward(q)
        position_error = target[:3, 3] - current_pose[:3, 3]
        orientation_error = rotation_vector(target[:3, :3] @ current_pose[:3, :3].T)
        error = np.concatenate((position_error, orientation_error))
        weights = np.ones(6, dtype=np.float64)
        if np.linalg.norm(error[3:]) > self.settings.orientation_hold_rad:
            weights[3:] = 0.0
        else:
            weights[3:] = self.settings.orientation_weight
        jacobian = np.asarray(self.kinematics.jacobian(q), dtype=np.float64)
        jacobian = jacobian.reshape(6, self.rest.size)
        weighted_jacobian = jacobian * weights[:, None]
        singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
        smallest = float(np.min(singular_values))
        ramp = max(0.0, 1.0 - smallest / self.settings.singular_value_threshold)
        damping_sq = self.settings.base_damping**2 + (self.settings.singularity_damping * ramp) ** 2
        rest_sq = self.settings.rest_gain**2
        normal = weighted_jacobian.T @ weighted_jacobian
        normal += (damping_sq + rest_sq) * np.eye(self.rest.size)
        rhs = weighted_jacobian.T @ (error * weights) + rest_sq * (self.rest - q)
        candidate = np.clip(q + np.linalg.solve(normal, rhs), self.lower, self.upper)
        maximum_dt = self._nominal_dt_s * self.settings.max_step_multiplier
        bounded_dt = self._nominal_dt_s
        if math.isfinite(dt_s) and dt_s > 0.0:
            bounded_dt = min(dt_s, maximum_dt)
        change = np.clip(candidate - q, -self.speed * bounded_dt, self.speed * bounded_dt)
        return np.clip(q + change, self.lower, self.upper)
