from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robotics_stack.robots.piper import JOINT_SIGNS, PiperConnection, _Arm
from robotics_stack.robots.piper_kinematics import PiperArmKinematics


class _Sdk:
    def __init__(self) -> None:
        self.joint_command: tuple[int, ...] | None = None
        self.gripper_command: int | None = None
        self.disabled = False
        self.disconnected = False

    def JointCtrl(self, *values: int) -> None:
        self.joint_command = values

    def GripperCtrl(self, position: int, *_: int) -> None:
        self.gripper_command = position

    def DisablePiper(self) -> None:
        self.disabled = True

    def DisconnectPort(self) -> None:
        self.disconnected = True

    def GetArmJointMsgs(self):
        return SimpleNamespace(
            time_stamp=1.0,
            joint_state=SimpleNamespace(**{f"joint_{i}": i * 1000 for i in range(1, 7)}),
        )


def _arm() -> _Arm:
    arm = object.__new__(_Arm)
    arm.port = "test-can"
    arm._port_open = True
    arm.config = type(
        "Config",
        (),
        {
            "action_space": "normalized_100",
            "gripper_effort": 1000,
            "disable_on_disconnect": True,
        },
    )()
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


def test_disconnect_disables_motors_before_closing_can() -> None:
    arm = _arm()
    arm.close()
    assert arm.sdk.disabled is True
    assert arm.sdk.disconnected is True


def test_default_disconnect_latches_feedback_without_disabling_motors() -> None:
    arm = _arm()
    arm.config = PiperConnection()
    arm.close()
    assert arm.sdk.joint_command == (1000, 2000, 3000, 4000, 5000, 6000)
    assert arm.sdk.disabled is False
    assert arm.sdk.disconnected is True
    arm.sdk.joint_command = None
    arm.close()
    assert arm.sdk.joint_command is None  # Repeated cleanup sends nothing.


def test_connection_never_automatically_resets_a_faulted_arm() -> None:
    import pytest

    arm = _arm()
    arm.config = PiperConnection()
    arm.sdk.ConnectPort = lambda: None
    arm.sdk.ResetPiper = lambda: pytest.fail("ResetPiper releases torque")
    arm._wait_for_feedback = lambda: None
    arm._requires_reset = lambda: True
    with pytest.raises(RuntimeError, match="automatic ResetPiper is disabled"):
        arm.connect()
    assert arm.sdk.disabled is False
    assert arm.sdk.disconnected is True


def test_link_positions_include_configured_base_and_tcp() -> None:
    model = PiperArmKinematics(lateral_offset_m=0.3)
    positions = model.link_positions(np.zeros(6))
    assert positions.shape == (8, 3)
    assert positions[0].tolist() == [0.0, 0.3, 0.0]
    assert np.allclose(positions[-1], model.forward(np.zeros(6))[:3, 3])


def test_radian_feedback_tolerates_small_stop_error_but_targets_remain_strict():
    from pathlib import Path
    from types import SimpleNamespace

    import pytest

    from robotics_stack.config.loaders import load_station_config
    from robotics_stack.contracts import ContractError
    from robotics_stack.robots.piper import BiPiper, PiperConnection

    root = Path(__file__).parents[1]
    schema, _, _ = load_station_config(root / "configs/robots/piper_bimanual_dream1.yaml")
    robot = object.__new__(BiPiper)
    robot.schema = schema
    robot.connection = PiperConnection(action_space="radians")
    robot.connected = True
    left = np.zeros(6)
    right = np.zeros(6)
    right[1] = -0.0020420352248333657
    robot.left = SimpleNamespace(joints=lambda: left, gripper=lambda: 0.0)
    robot.right = SimpleNamespace(joints=lambda: right, gripper=lambda: 0.0)
    assert robot.observation()[8] == 0.0
    assert right[1] < 0  # Raw feedback is retained.
    action = list(robot.observation())
    action[8] = right[1]
    with pytest.raises(ContractError, match="right_joint_2"):
        robot.set_target(tuple(action))
    right[1] = -0.01
    with pytest.raises(ContractError, match="right_joint_2"):
        robot.observation()
    right[1] = float("nan")
    with pytest.raises(ContractError, match="right_joint_2"):
        robot.observation()
