from robotics_stack.runtime.policy_agent import _trained_chunk_is_usable


def test_trained_rtc_rejects_a_result_that_outruns_its_conditioned_prefix() -> None:
    assert _trained_chunk_is_usable(
        conditioned_delay=10, measured_delay=10, training_max_delay=25, has_prefix=True
    )
    assert not _trained_chunk_is_usable(
        conditioned_delay=10, measured_delay=11, training_max_delay=25, has_prefix=True
    )


def test_initial_rtc_chunk_can_bootstrap_within_the_training_limit() -> None:
    assert _trained_chunk_is_usable(
        conditioned_delay=0, measured_delay=25, training_max_delay=25, has_prefix=False
    )
