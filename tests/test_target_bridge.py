import pytest

from robotics_stack.teleoperators.guidance.tempo import (
    FixedRateTargetBridge,
    TargetBridgeSettings,
    TargetTimeline,
)


def test_timeline_interpolates_between_timestamped_operator_targets() -> None:
    timeline = TargetTimeline(
        TargetBridgeSettings(
            input_rate_hz=10,
            output_rate_hz=100,
            playback_delay_s=0.05,
        )
    )
    timeline.reset((0.0, 0.0), timestamp_s=10.0)
    timeline.push((1.0, 2.0), timestamp_s=10.1)

    values, mode = timeline.sample(now_s=10.1) or ((), "missing")
    assert mode == "interpolate"
    assert values == pytest.approx((0.5, 1.0))


def test_bridge_holds_and_stops_when_operator_input_becomes_stale() -> None:
    delivered: list[tuple[float, ...]] = []
    holds: list[bool] = []
    bridge = FixedRateTargetBridge(
        delivered.append,
        lambda: holds.append(True),
        settings=TargetBridgeSettings(
            input_rate_hz=10,
            output_rate_hz=100,
            playback_delay_s=0.0,
            source_timeout_s=0.1,
        ),
    )
    bridge.submit((0.1, 0.2), timestamp_s=1.0, reset=True)
    assert bridge.tick(now_s=1.05)
    assert delivered == [(0.1, 0.2)]
    assert not bridge.tick(now_s=1.11)
    assert bridge.stale
    assert holds == [True]


def test_timeline_rejects_out_of_order_targets() -> None:
    timeline = TargetTimeline(TargetBridgeSettings())
    timeline.reset((0.0,), timestamp_s=1.0)
    with pytest.raises(ValueError, match="must increase"):
        timeline.push((0.1,), timestamp_s=1.0)


def test_bridge_timestamps_an_unstamped_target_on_the_local_clock(monkeypatch) -> None:
    bridge = FixedRateTargetBridge(
        lambda _values: None,
        lambda: None,
        settings=TargetBridgeSettings(),
    )
    clock = "robotics_stack.teleoperators.guidance.tempo.time.perf_counter"
    monkeypatch.setattr(clock, lambda: 42.0)

    bridge.submit((0.1,), reset=True)

    assert bridge.timeline.source_age_s(now_s=42.0) == 0.0
