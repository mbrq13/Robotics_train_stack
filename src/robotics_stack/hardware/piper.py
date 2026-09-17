"""Piper implementation for this stack's hardware contract.

The SDK stays isolated here. The station validates all targets before this
driver sees them and keeps both arms at their measured state when paused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robotics_stack.contracts import RobotSchema

RAD_TO_MDEG = 1000.0 * 180.0 / np.pi
MDEG_TO_RAD = 1.0 / RAD_TO_MDEG


@dataclass(frozen=True)
class PiperConnection:
    left_port: str = "can0"
    right_port: str = "can1"
    gripper_min_microm: int = 0
    gripper_max_microm: int = 70_000
    gripper_effort: int = 1000
    enable_timeout_s: float = 10.0
    feedback_timeout_s: float = 1.5
    startup_speed_percent: int = 10
    motion_speed_percent: int = 80
    gripper_calibration_file: str | None = None


def _read_gripper_calibration(path: str | None, side: str) -> tuple[int, int] | None:
    """Load explicit measured endpoints without silently changing other arm."""
    if path is None:
        return None
    source = Path(path).expanduser()
    try:
        document: Any = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        value = document.get(side, {})
        minimum = int(value["closed_microm"])
        maximum = int(value["open_microm"])
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid gripper calibration for {side}: {source}") from exc
    if maximum <= minimum:
        raise ValueError(f"gripper calibration for {side} must have positive travel")
    return minimum, maximum


class _Arm:
    def __init__(self, port: str, side: str, config: PiperConnection):
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError as exc:
            raise RuntimeError("install the station extra to use Piper hardware") from exc
        self.port = port
        self.side = side
        self.config = config
        self.sdk = C_PiperInterface_V2(port)
        self.gripper_min = config.gripper_min_microm
        self.gripper_max = config.gripper_max_microm

    def connect(self) -> None:
        self.sdk.ConnectPort()
        try:
            self._wait_for_feedback()
            measured = self.joints()
            if self._requires_reset():
                reset = getattr(self.sdk, "ResetPiper", None)
                if reset is not None:
                    reset()
                    time.sleep(0.1)
                    self._wait_for_feedback()

            # Piper can retain a target from a previous process.  Seed its
            # controller with measured joints before changing mode or enabling
            # motors, so connection itself cannot send a go-home command.
            self._send_joints(measured)
            self._select_joint_mode(self.config.startup_speed_percent)
            self._send_joints(measured)
            self._enable()
            self._send_joints(measured)
            time.sleep(0.05)
            self._select_joint_mode(self.config.motion_speed_percent)
            self._send_joints(measured)
            self._load_gripper_range()
        except BaseException:
            self.close()
            raise

    def _wait_for_feedback(self) -> None:
        deadline = time.monotonic() + self.config.feedback_timeout_s
        while time.monotonic() < deadline:
            message = self.sdk.GetArmJointMsgs()
            if float(getattr(message, "time_stamp", 0.0)) > 0.0:
                return
            time.sleep(0.02)
        raise TimeoutError(
            f"{self.port}: no Piper feedback; verify power, CAN mapping and bitrate"
        )

    def _requires_reset(self) -> bool:
        status = getattr(self.sdk.GetArmStatus(), "arm_status", None)
        return int(getattr(status, "ctrl_mode", 0)) == 0x02 or int(
            getattr(status, "arm_status", 0)
        ) != 0

    def _select_joint_mode(self, speed_percent: int) -> None:
        if not 1 <= int(speed_percent) <= 100:
            raise ValueError("Piper motion speed percentage must be in [1, 100]")
        self.sdk.MotionCtrl_2(0x01, 0x01, int(speed_percent), 0x00)

    def _enable(self) -> None:
        deadline = time.monotonic() + self.config.enable_timeout_s
        while not self.sdk.EnablePiper():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{self.port}: Piper did not enable")
            time.sleep(0.02)

    def _send_joints(self, joints: np.ndarray) -> None:
        encoded = np.rint(joints * RAD_TO_MDEG).astype(np.int64)
        self.sdk.JointCtrl(*(int(value) for value in encoded))

    def _load_gripper_range(self) -> None:
        configured = _read_gripper_calibration(self.config.gripper_calibration_file, self.side)
        if configured is not None:
            self.gripper_min, self.gripper_max = configured
            return
        minimum, maximum = self.sdk.GetSDKGripperRangeParam()
        self.gripper_min = int(round(float(minimum) * 1_000_000))
        self.gripper_max = int(round(float(maximum) * 1_000_000))
        if self.gripper_max <= self.gripper_min:
            raise RuntimeError(f"{self.port}: invalid Piper gripper range")

    def close(self) -> None:
        close = getattr(self.sdk, "DisconnectPort", None)
        if close is not None:
            close()

    def joints(self) -> np.ndarray:
        joints = self.sdk.GetArmJointMsgs().joint_state
        return (
            np.asarray(
                [
                    joints.joint_1,
                    joints.joint_2,
                    joints.joint_3,
                    joints.joint_4,
                    joints.joint_5,
                    joints.joint_6,
                ],
                dtype=np.float64,
            )
            * MDEG_TO_RAD
        )

    def gripper(self) -> float:
        message = self.sdk.GetArmGripperMsgs().gripper_state
        span = max(1, self.gripper_max - self.gripper_min)
        return float(np.clip((message.grippers_angle - self.gripper_min) / span, 0.0, 1.0))

    def target(self, joints: np.ndarray, gripper: float) -> None:
        self._send_joints(joints)
        opening = float(np.clip(gripper, 0.0, 1.0))
        position = int(round(self.gripper_min + opening * (self.gripper_max - self.gripper_min)))
        self.sdk.GripperCtrl(position, self.config.gripper_effort, 0x01, 0x00)


class BiPiper:
    """Two six-axis Piper arms with 14D state/action ordering: left then right."""

    def __init__(self, schema: RobotSchema, connection: PiperConnection):
        if schema.action_size != 14:
            raise ValueError("BiPiper requires a 14-dimensional action schema")
        self.schema = schema
        self.connection = connection
        self.left = _Arm(connection.left_port, "left", connection)
        self.right = _Arm(connection.right_port, "right", connection)
        self.connected = False

    def connect(self) -> None:
        try:
            self.left.connect()
            self.right.connect()
            self.connected = True
            self.hold()
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        self.connected = False
        for arm in (self.right, self.left):
            try:
                arm.close()
            except Exception:
                pass

    def observation(self) -> tuple[float, ...]:
        if not self.connected:
            raise RuntimeError("Piper is disconnected")
        return tuple(
            [*self.left.joints(), self.left.gripper(), *self.right.joints(), self.right.gripper()]
        )

    def set_target(self, action: tuple[float, ...]) -> None:
        values = self.schema.validate_action(action)
        self.left.target(values[:6], float(values[6]))
        self.right.target(values[7:13], float(values[13]))

    def hold(self) -> None:
        self.set_target(self.observation())

    def home(self) -> None:
        self.set_target(tuple(0.0 for _ in range(14)))
