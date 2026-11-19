from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robotics_stack.contracts import JointLimit, RobotSchema
from robotics_stack.robots import so100


def _schema() -> RobotSchema:
    limits = tuple(
        JointLimit(-3.0, 3.0, 0.5, 1.0) for _ in range(5)
    ) + (JointLimit(0.0, 1.0, 0.5, 1.0),)
    return RobotSchema(
        name="so100-test",
        action_names=so100.SO100_NAMES,
        state_names=so100.SO100_NAMES,
        joint_limits=limits,
        action_space="radians",
        robot_type="so100",
        control_hz=20.0,
    )


class FakeArm:
    def __init__(self) -> None:
        self.positions = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, 0.0])
        self.gripper = 0.75
        self.commands = []
        self.held = 0
        self.closed = False
        self.powered_off = False

    def read(self):
        return SimpleNamespace(positions_rad=self.positions, gripper=self.gripper)

    def write(self, command):
        self.commands.append(command)

    def hold(self):
        self.held += 1

    def close(self):
        self.closed = True

    def torque_off(self):
        self.powered_off = True


def test_single_so100_maps_five_joints_and_one_gripper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = tmp_path / "arm.json"
    calibration.write_text("{}", encoding="utf-8")
    fake = FakeArm()
    monkeypatch.setattr(so100, "_open_follower", lambda connection: fake)
    driver = so100.SO100Single(
        _schema(),
        so100.SO100Connection(
            port="/dev/serial/by-id/test-arm",
            calibration_file=str(calibration),
            home=(0.0, -0.8, 0.7, 1.1, 0.0),
        ),
    )

    driver.connect()
    assert driver.observation() == pytest.approx((0.1, -0.2, 0.3, -0.4, 0.5, 0.75))
    driver.set_target((0.2, -0.1, 0.4, -0.3, 0.6, 0.25))
    assert fake.commands[-1].positions_rad.tolist() == pytest.approx(
        [0.2, -0.1, 0.4, -0.3, 0.6, 0.0]
    )
    assert fake.commands[-1].gripper == pytest.approx(0.25)
    assert fake.held == 1

    driver.disconnect()
    assert fake.powered_off is True


def test_so100_refuses_an_unstable_serial_name(tmp_path) -> None:
    calibration = tmp_path / "arm.json"
    calibration.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="by-id"):
        so100.SO100Single(
            _schema(),
            so100.SO100Connection(
                port="/dev/ttyUSB0",
                calibration_file=str(calibration),
                home=(0.0, -0.8, 0.7, 1.1, 0.0),
            ),
        )
