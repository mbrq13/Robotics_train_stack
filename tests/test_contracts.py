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


def test_supervisor_uses_a_separate_bounded_age_for_rtc_actions() -> None:
    guard = MotionSupervisor(schema(), max_action_age_ms=100, max_scheduled_action_age_ms=600)
    guard.connected()
    session = guard.arm()
    guard.start()
    scheduled = PolicyAction(session, 1, 1, 1_000_000_000, 1, (0.25, 0.5), scheduled=True)
    assert guard.validate(scheduled, now_ns=1_500_000_000).accepted
    expired = PolicyAction(session, 2, 2, 1_000_000_000, 1, (0.25, 0.5), scheduled=True)
    assert not guard.validate(expired, now_ns=1_601_000_000).accepted


def test_supervisor_has_a_distinct_first_action_deadline() -> None:
    guard = MotionSupervisor(schema())
    guard.connected()
    guard.arm()
    guard.start()
    assert guard.run_started_at_ns is not None
    assert not guard.watchdog_expired(
        400,
        first_action_timeout_ms=15_000,
        now_ns=guard.run_started_at_ns + 14_999_000_000,
    )
    assert guard.watchdog_expired(
        400,
        first_action_timeout_ms=15_000,
        now_ns=guard.run_started_at_ns + 15_001_000_000,
    )
