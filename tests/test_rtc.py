from __future__ import annotations

import numpy as np
import pytest

from robotics_stack.learning.pi05 import pi05_train_command
from robotics_stack.policy.rtc import RtcActionQueue, RtcChunk, RtcSettings


def _settings() -> RtcSettings:
    return RtcSettings(
        control_hz=30,
        training_max_delay=2,
        execution_horizon=3,
        refill_threshold=3,
    )


def _chunk(offset: float = 0) -> RtcChunk:
    actions = np.asarray([[offset + index, offset + index + 0.5] for index in range(6)])
    return RtcChunk(raw=actions, actions=actions + 100)


def test_rtc_queue_preserves_in_flight_prefix_and_skips_conditioned_chunk_prefix() -> None:
    queue = RtcActionQueue(_settings(), action_dim=2)
    queue.merge(
        _chunk(),
        inference_delay_steps=0,
        observation_id=1,
        station_monotonic_ns=10,
        control_generation=4,
    )
    first = queue.pop()
    assert first is not None
    assert tuple(first.action) == (100, 100.5)
    assert first.control_generation == 4

    queue.merge(
        _chunk(20),
        inference_delay_steps=2,
        observation_id=2,
        station_monotonic_ns=20,
        control_generation=5,
    )
    assert len(queue) == 5  # two previously queued + three new planned actions
    preserved = queue.pop()
    assert preserved is not None
    assert tuple(preserved.action) == (101, 101.5)
    assert preserved.control_generation == 4
    first_new = list(queue._items)[1]
    assert tuple(first_new.action) == (122, 122.5)
    assert first_new.control_generation == 5


def test_rtc_settings_rejects_horizon_without_training_margin() -> None:
    settings = _settings()
    with pytest.raises(ValueError, match="chunk_size"):
        settings.validate_chunk_size(4)


def test_pi05_rtc_launcher_validates_contract(tmp_path) -> None:
    config = tmp_path / "pi05.yaml"
    config.write_text(
        "policy:\n  type: pi05\n  chunk_size: 50\n  rtc_training_max_delay: 10\n",
        encoding="utf-8",
    )
    assert pi05_train_command(config) == ["lerobot-train", "--config_path", str(config)]
