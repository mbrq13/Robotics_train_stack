"""VR-to-joint teleoperation with explicit per-arm clutching."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from robotics_stack.contracts import RobotSchema
from robotics_stack.robots.piper_kinematics import (
    PIPER_LOWER_RAD,
    PIPER_UPPER_RAD,
    PiperArmKinematics,
)
from robotics_stack.teleoperators.vr.dls import DampedLeastSquares, DlsSettings
from robotics_stack.teleoperators.vr.motion import OneEuroJointFilter
from robotics_stack.teleoperators.vr.tracking import ControllerSample, TrackingFrame
from robotics_stack.teleoperators.vr.transforms import anchored_target, matrix_pose


@dataclass(frozen=True)
class VrTeleoperatorSettings:
    """Rig-qualified VR control parameters, intentionally separate from a policy."""

    tracking_rate_hz: float = 72.0
    joint_speed_cap_rad_s: float = 1.0
    translation_scale: float = 1.0
    max_displacement_m: float = 0.60
    gripper_rate_per_s: float = 0.75
    tracking_timeout_s: float = 0.15
    source_to_robot_rotation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        values = (
            self.tracking_rate_hz,
            self.joint_speed_cap_rad_s,
            self.translation_scale,
            self.max_displacement_m,
            self.gripper_rate_per_s,
            self.tracking_timeout_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("VR rates, scale and timeouts must be positive")
        matrix = np.asarray(self.source_to_robot_rotation, dtype=np.float64)
        if matrix.shape != (9,) or not np.isfinite(matrix).all():
            raise ValueError("VR source_to_robot_rotation must contain nine finite values")


class VrTeleoperator:
    """Convert tracked controller poses into bounded Piper joint targets.

    Arms are deliberately dormant until ``engage`` is called. Engagement anchors
    the current controller pose to the current robot tool pose, avoiding a jump
    when an operator begins a correction or resumes after tracking loss.
    """

    def __init__(
        self, schema: RobotSchema, *, settings: VrTeleoperatorSettings | None = None
    ) -> None:
        self.schema = schema
        self.settings = settings or VrTeleoperatorSettings()
        self._validate_schema()
        self._kinematics = {"left": PiperArmKinematics(), "right": PiperArmKinematics()}
        self._solvers = {
            side: DampedLeastSquares(
                self._kinematics[side],
                rest_joints=np.zeros(6),
                lower=PIPER_LOWER_RAD,
                upper=PIPER_UPPER_RAD,
                max_speed_rad_s=np.full(6, self.settings.joint_speed_cap_rad_s),
                nominal_rate_hz=self.settings.tracking_rate_hz,
                settings=DlsSettings(),
            )
            for side in ("left", "right")
        }
        self._state: np.ndarray | None = None
        self._anchors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_timestamp_ns: int | None = None
        self._filter = OneEuroJointFilter()

    def set_state(self, values: tuple[float, ...]) -> None:
        """Synchronize from measured robot state before beginning an epoch."""
        state = self.schema.validate_action(values)
        self._state = state.copy()
        self._anchors.clear()
        self._last_timestamp_ns = None
        self._filter.reset(state)

    @property
    def target(self) -> tuple[float, ...] | None:
        """Most recent bounded target, suitable for recording after submission."""
        if self._state is None:
            return None
        return tuple(float(value) for value in self._state)

    def engage(self, frame: TrackingFrame, *, sides: tuple[str, ...] = ("left", "right")) -> None:
        """Anchor selected tracked controllers to the current measured tools."""
        state = self._require_state()
        for side in sides:
            controller = self._controller(frame, side)
            if not controller.tracked:
                raise RuntimeError(f"cannot engage {side} without valid VR tracking")
            joints = state[self._joint_slice(side)]
            tool = matrix_pose(self._kinematics[side].forward(joints))
            self._anchors[side] = (np.asarray(controller.pose), tool)
        self._last_timestamp_ns = frame.timestamp_ns

    def disengage(self, *sides: str) -> None:
        """Drop anchors so stale or returned tracking cannot generate a jump."""
        targets = sides or tuple(self._anchors)
        for side in targets:
            self._anchors.pop(side, None)

    def step(self, frame: TrackingFrame, *, now_ns: int | None = None) -> tuple[float, ...] | None:
        """Advance active arms by one DLS step, or return ``None`` on stale input."""
        if self._state is None or not self._anchors:
            return None
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if now_ns - frame.timestamp_ns > int(self.settings.tracking_timeout_s * 1_000_000_000):
            self.disengage()
            return None
        previous_ns = self._last_timestamp_ns or frame.timestamp_ns
        dt_s = max(0.0, (frame.timestamp_ns - previous_ns) / 1_000_000_000)
        rotation = np.asarray(self.settings.source_to_robot_rotation, dtype=np.float64)
        rotation = rotation.reshape(3, 3)
        next_state = self._state.copy()
        for side, (source_anchor, robot_anchor) in tuple(self._anchors.items()):
            controller = self._controller(frame, side)
            if not controller.tracked:
                self.disengage(side)
                continue
            target = anchored_target(
                source_anchor,
                np.asarray(controller.pose),
                robot_anchor,
                source_to_robot_rotation=rotation,
                translation_scale=self.settings.translation_scale,
                max_displacement_m=self.settings.max_displacement_m,
            )
            joint_slice = self._joint_slice(side)
            next_joints = self._solvers[side].step(next_state[joint_slice], target, dt_s=dt_s)
            next_state[joint_slice] = next_joints
            gripper_index = 6 if side == "left" else 13
            next_state[gripper_index] = np.clip(
                next_state[gripper_index]
                + controller.stick_y * self.settings.gripper_rate_per_s * dt_s,
                self.schema.joint_limits[gripper_index].minimum,
                self.schema.joint_limits[gripper_index].maximum,
            )
        self._state = self.schema.validate_action(next_state)
        self._last_timestamp_ns = frame.timestamp_ns
        self._state = self.schema.validate_action(
            self._filter.apply(self._state, timestamp_ns=frame.timestamp_ns)
        )
        return tuple(float(value) for value in self._state)

    def _validate_schema(self) -> None:
        expected = (
            *(f"left_joint_{index}" for index in range(1, 7)),
            "left_gripper",
            *(f"right_joint_{index}" for index in range(1, 7)),
            "right_gripper",
        )
        if self.schema.state_names != expected or self.schema.action_names != expected:
            raise ValueError(
                "the bundled VR Piper profile requires the canonical 14-axis Piper schema"
            )

    def _require_state(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("set measured robot state before engaging VR teleoperation")
        return self._state

    @staticmethod
    def _joint_slice(side: str) -> slice:
        if side == "left":
            return slice(0, 6)
        if side == "right":
            return slice(7, 13)
        raise ValueError(f"unknown Piper side: {side!r}")

    @staticmethod
    def _controller(frame: TrackingFrame, side: str) -> ControllerSample:
        if side == "left":
            return frame.left
        if side == "right":
            return frame.right
        raise ValueError(f"unknown Piper side: {side!r}")
