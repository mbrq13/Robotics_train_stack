from robotics_stack.contracts import JointLimit, PolicyAction, RobotSchema
from robotics_stack.control.supervisor import MotionSupervisor, RunState


def schema() -> RobotSchema:
    return RobotSchema(
        name="test",
        action_names=("joint", "gripper"),
        state_names=("joint", "gripper"),
        joint_limits=(JointLimit(-1, 1, 1, 1), JointLimit(0, 1, 1, 1)),
    )


def action(session: str, sequence: int = 1, timestamp: int = 1_000_000_000) -> PolicyAction:
    return PolicyAction(session, sequence, 1, timestamp, 1, (0.25, 0.5))


def test_supervisor_requires_arm_start_and_current_session() -> None:
    guard = MotionSupervisor(schema(), max_action_age_ms=100)
    guard.connected()
    session = guard.arm()
    assert guard.state is RunState.ARMED
    assert not guard.validate(action(session), now_ns=1_000_000_010).accepted
    guard.start()
    assert guard.validate(action(session), now_ns=1_000_000_010).accepted
    assert not guard.validate(action(session, sequence=1), now_ns=1_000_000_020).accepted
    assert not guard.validate(action("another", sequence=2), now_ns=1_000_000_030).accepted


def test_supervisor_rejects_expired_or_out_of_range_actions() -> None:
    guard = MotionSupervisor(schema(), max_action_age_ms=1)
    guard.connected()
    session = guard.arm()
    guard.start()
    assert not guard.validate(action(session), now_ns=1_002_000_000).accepted
    invalid = PolicyAction(session, 2, 2, 1_002_000_000, 1, (4.0, 0.5))
    assert not guard.validate(invalid, now_ns=1_002_000_001).accepted
