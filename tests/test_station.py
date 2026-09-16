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
