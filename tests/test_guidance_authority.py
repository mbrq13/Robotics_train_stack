import pytest

from robotics_stack.guidance.authority import ControlAuthority, ControlSource, GuidancePhase


def test_authority_moves_between_policy_hold_and_operator_control() -> None:
    authority = ControlAuthority()

    assert authority.activate_policy().generation == 0
    assert authority.snapshot.phase is GuidancePhase.POLICY_RUN
    assert authority.accepts(ControlSource.POLICY, 0)

    held = authority.hold()
    assert held.phase is GuidancePhase.HOLDING
    assert held.generation == 1

    correction = authority.begin_correction()
    assert correction.source is ControlSource.OPERATOR
    assert correction.generation == 2

    authority.finish_operator_control()
    resumed = authority.activate_policy()
    assert resumed.phase is GuidancePhase.POLICY_RUN
    assert resumed.generation == 3
    assert not authority.accepts(ControlSource.POLICY, 0)


def test_recovery_requires_a_physical_hold_first() -> None:
    authority = ControlAuthority()
    authority.activate_policy()

    with pytest.raises(RuntimeError, match="holding"):
        authority.begin_recovery()

    authority.hold()
    assert authority.begin_recovery().phase is GuidancePhase.RECOVERY
