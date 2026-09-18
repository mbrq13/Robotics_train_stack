import pytest

from robotics_stack.evaluation.latency import summarize_latency


def test_latency_summary_uses_interpolated_percentiles_and_control_steps() -> None:
    summary = summarize_latency([0.010, 0.020, 0.030, 0.040], control_hz=30)

    assert summary.samples == 4
    assert summary.p50_ms == 25.0
    assert summary.p95_ms == pytest.approx(38.5)
    assert summary.delay_steps_p95 == 2


def test_latency_summary_rejects_invalid_measurements() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        summarize_latency([0.001, -0.001], control_hz=30)
