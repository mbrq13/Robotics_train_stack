from __future__ import annotations

import numpy as np

from robotics_stack.hardware.piper import JOINT_SIGNS, _Arm


class _Sdk:
    def __init__(self) -> None:
        self.joint_command: tuple[int, ...] | None = None
        self.gripper_command: int | None = None

    def JointCtrl(self, *values: int) -> None:
        self.joint_command = values

    def GripperCtrl(self, position: int, *_: int) -> None:
        self.gripper_command = position


def _arm() -> _Arm:
    arm = object.__new__(_Arm)
    arm.config = type("Config", (), {"action_space": "normalized_100", "gripper_effort": 1000})()
    arm.sdk = _Sdk()
    arm.joint_min_deg = np.asarray((-150, 0, -170, -100, -70, -120), dtype=np.float64)
    arm.joint_max_deg = np.asarray((150, 180, 0, 100, 70, 120), dtype=np.float64)
    arm.gripper_min = 0
    arm.gripper_max = 70_000
    return arm


def test_normalized_midpoint_maps_to_each_physical_joint_midpoint() -> None:
    arm = _arm()
    arm.target(np.zeros(6), 50.0)

    expected_degrees = (arm.joint_min_deg + arm.joint_max_deg) / 2.0
    expected_mdeg = np.rint(np.deg2rad(JOINT_SIGNS * expected_degrees) * (1000 * 180 / np.pi))
    assert arm.sdk.joint_command == tuple(int(value) for value in expected_mdeg)
    assert arm.sdk.gripper_command == 35_000


def test_normalized_endpoints_map_to_physical_limits() -> None:
    arm = _arm()
    arm.target(np.full(6, -100.0), 0.0)
    minimum = np.rint(np.deg2rad(JOINT_SIGNS * arm.joint_min_deg) * (1000 * 180 / np.pi))
    assert arm.sdk.joint_command == tuple(int(value) for value in minimum)
    assert arm.sdk.gripper_command == 0

    arm.target(np.full(6, 100.0), 100.0)
    maximum = np.rint(np.deg2rad(JOINT_SIGNS * arm.joint_max_deg) * (1000 * 180 / np.pi))
    assert arm.sdk.joint_command == tuple(int(value) for value in maximum)
    assert arm.sdk.gripper_command == 70_000
