import pytest

from robotics_stack.rollout.smoothing import ExponentialActionSmoother


def test_identity_alpha_preserves_checkpoint_actions() -> None:
    smoother = ExponentialActionSmoother()
    assert smoother.apply((1.0, -2.0)) == (1.0, -2.0)
    assert smoother.apply((-3.0, 4.0)) == (-3.0, 4.0)


def test_smoother_uses_previous_emitted_action_and_resets() -> None:
    smoother = ExponentialActionSmoother(0.25)
    assert smoother.apply((0.0, 8.0)) == (0.0, 8.0)
    assert smoother.apply((4.0, 0.0)) == (1.0, 6.0)
    smoother.reset()
    assert smoother.apply((4.0, 0.0)) == (4.0, 0.0)


@pytest.mark.parametrize("alpha", (0.0, -0.1, 1.1))
def test_smoother_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="must be in"):
        ExponentialActionSmoother(alpha)
