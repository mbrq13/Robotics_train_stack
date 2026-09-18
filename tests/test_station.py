import pytest

from robotics_stack.contracts import JointLimit, PolicyAction, RobotSchema
from robotics_stack.hardware.fake import FakeRobot
from robotics_stack.runtime.station import RobotStation


def schema() -> RobotSchema:
    return RobotSchema(
        name="fake",
        action_names=("joint", "gripper"),
        state_names=("joint", "gripper"),
        joint_limits=(JointLimit(-1, 1, 1, 1), JointLimit(0, 1, 1, 1)),
    )


def test_station_only_moves_after_accepted_action() -> None:
    robot = FakeRobot(schema())
    station = RobotStation(robot, schema())
    station.connect()
    session = station.arm()
    station.start()
    incoming = PolicyAction(session, 1, 1, 1, 1, (0.5, 0.4))
    verdict = station.supervisor.validate(incoming, now_ns=2)
    assert verdict.accepted
    assert station.status().state == "running"
    station.pause()
    assert station.status().state == "paused"
    station.disconnect()


def test_station_operator_handoff_rejects_stale_targets_and_resumes_fresh_session() -> None:
    robot = FakeRobot(schema())
    station = RobotStation(robot, schema())
    station.connect()
    first_session = station.arm()
    station.start()

    generation = station.begin_operator_correction()
    assert station.status().state == "paused"
    assert station.status().control_phase == "correction"
    with pytest.raises(RuntimeError, match="inactive control generation"):
        station.apply_operator_target((0.4, 0.5), control_generation=generation - 1)

    station.apply_operator_target((0.4, 0.5), control_generation=generation)
    assert robot.observation() == (0.4, 0.5)
    station.finish_operator_control()
    second_session = station.resume_policy()

    assert second_session != first_session
    assert station.status().state == "running"
    assert station.status().control_phase == "policy_run"
    station.disconnect()
