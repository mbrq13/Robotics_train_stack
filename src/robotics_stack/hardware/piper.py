"""Piper implementation for this stack's hardware contract.

The SDK stays isolated here. The station validates all targets before this
driver sees them and keeps both arms at their measured state when paused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

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


class _Arm:
    def __init__(self, port: str, config: PiperConnection):
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError as exc:
            raise RuntimeError("install the station extra to use Piper hardware") from exc
        self.port = port
        self.config = config
        self.sdk = C_PiperInterface_V2(port)
        self.gripper_min = config.gripper_min_microm
        self.gripper_max = config.gripper_max_microm

    def connect(self) -> None:
        self.sdk.ConnectPort()
        deadline = time.monotonic() + self.config.enable_timeout_s
        while not self.sdk.EnablePiper():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{self.port}: Piper did not enable")
            time.sleep(0.02)
        try:
            minimum, maximum = self.sdk.GetSDKGripperRangeParam()
            self.gripper_min = int(round(float(minimum) * 1_000_000))
            self.gripper_max = int(round(float(maximum) * 1_000_000))
        except Exception:
            # Explicit configured calibration remains a safe fallback.
            pass

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
        encoded = np.rint(joints * RAD_TO_MDEG).astype(np.int64)
        self.sdk.JointCtrl(*(int(value) for value in encoded))
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
        self.left = _Arm(connection.left_port, connection)
        self.right = _Arm(connection.right_port, connection)
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
