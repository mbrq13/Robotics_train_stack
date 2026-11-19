"""Single physical SO-100 adapter for Dream inference.

The arm exposes five joints in radians and one normalized gripper.  Its six
STS3215 servos share one serial bus.  The low-level bus implementation comes
from the Dream Labs hardware package so calibration, synchronized I/O,
temperature guards and torque-off behaviour stay aligned with its physical
robot flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotics_stack.contracts import ContractError, RobotSchema
from robotics_stack.robots.registry import register_robot_driver

SO100_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class SO100Connection:
    port: str
    calibration_file: str
    home: tuple[float, float, float, float, float]
    disable_on_disconnect: bool = True

    def validate(self) -> None:
        if not self.port.startswith(("/dev/serial/by-id/", "/dev/serial/by-path/")):
            raise ValueError(
                "SO-100 port must use /dev/serial/by-id or /dev/serial/by-path; "
                "a ttyUSB/ttyACM number can change after reconnect"
            )
        if len(self.home) != 5 or not np.isfinite(self.home).all():
            raise ValueError("SO-100 home must contain five finite joint angles in radians")
        calibration = Path(self.calibration_file).expanduser()
        if not calibration.is_file():
            raise ValueError(f"SO-100 calibration file does not exist: {calibration}")


def _open_follower(connection: SO100Connection) -> Any:
    """Open the calibrated arm without importing hardware dependencies at startup."""
    try:
        from robots import config as dream_config
        from robots import feetech, motor_arm, so101
    except ImportError as exc:
        raise RuntimeError(
            "install the so100 optional dependency to use the physical arm"
        ) from exc

    calibration = str(Path(connection.calibration_file).expanduser().resolve())
    try:
        open_deg, closed_deg = so101.gripper_ends(calibration)
        arm_config = dream_config.ArmConfig(
            can_channel=connection.port,
            calibration=calibration,
            home=(*connection.home, 0.0),
            gripper_open_deg=open_deg,
            gripper_closed_deg=closed_deg,
        )
        return motor_arm.open_arm(
            so101.ARM,
            arm_config,
            "right",
            opener=feetech.open_bus,
            energize=True,
        )
    except ImportError as exc:
        raise RuntimeError(
            "the SO-100 servo bus is unavailable; install the so100 optional dependency"
        ) from exc


class SO100Single:
    """One physical arm; no left-arm device or action exists in this contract."""

    def __init__(self, schema: RobotSchema, connection: SO100Connection):
        if schema.robot_type != "so100":
            raise ContractError("SO100Single requires schema.robot_type=so100")
        if schema.action_space != "radians":
            raise ContractError("SO100Single requires radians plus a normalized gripper")
        if schema.action_names != SO100_NAMES or schema.state_names != SO100_NAMES:
            raise ContractError("SO100Single requires the canonical five-joint plus gripper schema")
        connection.validate()
        self.schema = schema
        self.connection = connection
        self._arm: Any | None = None

    def connect(self) -> None:
        if self._arm is not None:
            return
        arm = _open_follower(self.connection)
        try:
            # Opening energizes each servo at its measured position.  Hold that
            # pose explicitly; connection must never imply a move to home.
            arm.hold()
            reading = arm.read()
            if np.asarray(reading.positions_rad).shape != (6,):
                raise RuntimeError("SO-100 driver did not expose five joints and one dummy slot")
        except BaseException:
            try:
                arm.torque_off()
            finally:
                raise
        self._arm = arm

    def disconnect(self) -> None:
        arm, self._arm = self._arm, None
        if arm is None:
            return
        if self.connection.disable_on_disconnect:
            arm.torque_off()
        else:
            arm.close()

    def _connected(self) -> Any:
        if self._arm is None:
            raise RuntimeError("SO-100 is disconnected")
        return self._arm

    def observation(self) -> tuple[float, ...]:
        reading = self._connected().read()
        joints = np.asarray(reading.positions_rad, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError("SO-100 returned an invalid joint observation")
        gripper = float(np.clip(reading.gripper, 0.0, 1.0))
        return tuple(float(value) for value in (*joints[:5], gripper))

    def set_target(self, action: tuple[float, ...]) -> None:
        values = self.schema.validate_action(action)
        from robots.types import ArmCommand

        # Dream's SO arm model has a constant zero sixth joint.  The real arm
        # has five arm servos; servo 6 is the gripper and is passed separately.
        joints = np.concatenate((values[:5], np.zeros(1, dtype=np.float64)))
        self._connected().write(
            ArmCommand(positions_rad=joints, gripper=float(values[5]))
        )

    def hold(self) -> None:
        self._connected().hold()

    def home(self) -> None:
        self.set_target((*self.connection.home, 1.0))


def _build_so100(schema: RobotSchema, options: dict[str, Any]) -> SO100Single:
    raw = dict(options)
    raw["home"] = tuple(float(value) for value in raw.get("home", ()))
    connection = SO100Connection(**raw)
    return SO100Single(schema, connection)


register_robot_driver("so100_single", _build_so100)
