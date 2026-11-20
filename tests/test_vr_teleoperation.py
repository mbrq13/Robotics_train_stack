import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robotics_stack.config.loaders import load_station_config
from robotics_stack.kinematics.dls import DlsSettings
from robotics_stack.robots.fake import FakeRobot
from robotics_stack.services.station import RobotStation
from robotics_stack.teleoperators.vr.guided import GuidedVrHandoff
from robotics_stack.teleoperators.vr.session import VrTeleoperator, VrTeleoperatorSettings
from robotics_stack.teleoperators.vr.tracking import (
    ControllerSample,
    TrackingFrame,
    parse_quest_frame,
)


def _frame(*, timestamp_ns: int, left_x: float = 0.0) -> TrackingFrame:
    left = ControllerSample((left_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), tracked=True)
    right = ControllerSample((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), tracked=True)
    return TrackingFrame(timestamp_ns, left, right)


def test_quest_payload_is_normalized_into_controller_samples() -> None:
    frame = parse_quest_frame(
        {
            "leftControllerPosition": {"x": 1, "y": 2, "z": 3},
            "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "leftTracked": True,
            "leftValid": True,
            "buttonXPressed": True,
            "rightControllerPosition": [0, 0, 0],
            "rightControllerRotation": [0, 0, 0, 1],
            "rightTracked": True,
        },
        received_ns=123,
    )
    assert frame.timestamp_ns == 123
    assert frame.left.pose[:3] == (1.0, 2.0, 3.0)
    assert frame.left.primary_pressed
    assert frame.right.tracked


def test_piper_vr_clutch_returns_limited_targets_and_drops_stale_tracking() -> None:
    root = Path(__file__).parents[1]
    schema, _station, _profile = load_station_config(
        root / "configs" / "robots" / "piper_bimanual.yaml"
    )
    teleop = VrTeleoperator(schema)
    state = tuple(0.0 for _ in range(schema.action_size))
    teleop.set_state(state)
    teleop.engage(_frame(timestamp_ns=1_000_000_000), sides=("left",))
    action = teleop.step(_frame(timestamp_ns=1_013_888_889, left_x=0.10), now_ns=1_013_888_889)
    assert action is not None
    np.testing.assert_array_less(
        np.asarray(action), np.asarray([limit.maximum + 1e-9 for limit in schema.joint_limits])
    )
    assert teleop.step(_frame(timestamp_ns=1_013_888_889), now_ns=2_000_000_000) is None


def test_guided_vr_handoff_uses_station_authority_and_returns_to_hold() -> None:
    root = Path(__file__).parents[1]
    schema, _station, _profile = load_station_config(
        root / "configs" / "robots" / "piper_bimanual.yaml"
    )
    station = RobotStation(FakeRobot(schema), schema)
    station.connect()
    station.arm()
    station.start()
    station.pause()
    try:
        teleop = VrTeleoperator(schema)
        handoff = GuidedVrHandoff(station, teleop)
        timestamp = time.monotonic_ns()
        handoff.begin_correction(_frame(timestamp_ns=timestamp), sides=("left",))
        assert handoff.submit_tracking(_frame(timestamp_ns=timestamp + 1_000_000, left_x=0.01))
        handoff.finish_correction()
        assert station.status().control_phase == "holding"
    finally:
        station.disconnect()


def test_vr_solver_respects_station_speed_caps_and_configured_damping() -> None:
    root = Path(__file__).parents[1]
    schema, _, _ = load_station_config(root / "configs/robots/piper_bimanual.yaml")
    schema = replace(
        schema,
        joint_limits=tuple(replace(limit, max_speed=0.05) for limit in schema.joint_limits),
    )
    teleop = VrTeleoperator(schema, dls_settings=DlsSettings(rest_gain=0.0))
    teleop.set_state(tuple(0.0 for _ in schema.joint_limits))
    teleop.engage(_frame(timestamp_ns=1_000_000_000), sides=("left",))
    action = teleop.step(_frame(timestamp_ns=1_010_000_000, left_x=0.1), now_ns=1_010_000_000)
    assert action is not None
    assert np.max(np.abs(action[:6])) <= 0.05 * 0.01 + 1e-12
    assert teleop.step(
        _frame(timestamp_ns=1_010_000_000, left_x=0.2), now_ns=1_010_000_000
    ) is None
    assert teleop.step(
        _frame(timestamp_ns=1_005_000_000, left_x=0.2), now_ns=1_010_000_000
    ) is None
    assert teleop.target == action


@pytest.mark.parametrize("matrix", [[0.0] * 9, [-1, 0, 0, 0, 1, 0, 0, 0, 1]])
def test_vr_rejects_singular_and_reflected_frame_maps(matrix) -> None:
    with pytest.raises(ValueError, match="proper rotation"):
        VrTeleoperatorSettings(source_to_robot_rotation=tuple(matrix))


def test_stationary_controller_holds_measured_nonzero_posture() -> None:
    root = Path(__file__).parents[1]
    schema, _, _ = load_station_config(root / "configs/robots/piper_bimanual.yaml")
    teleop = VrTeleoperator(schema)
    state = (0.1, 0.8, -0.7, 0.2, 0.3, -0.2, 0.4) * 2
    teleop.set_state(state)
    teleop.engage(_frame(timestamp_ns=1_000_000_000))
    action = teleop.step(_frame(timestamp_ns=1_010_000_000), now_ns=1_010_000_000)
    assert np.allclose(action, state, atol=1e-12)
