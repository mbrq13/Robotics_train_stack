from pathlib import Path

import pytest

from robotics_stack.runtime.config import load_station_config


def test_default_piper_profile_uses_the_documented_motion_envelope() -> None:
    root = Path(__file__).parents[1]
    schema, station, _connection = load_station_config(root / "configs" / "piper_station.yaml")

    assert station["motion_mode"] == "bounded"
    joint_limits = [limit for index, limit in enumerate(schema.joint_limits) if index % 7 != 6]
    assert all(limit.max_speed == pytest.approx(3.141592653589793) for limit in joint_limits)
    assert all(
        limit.max_acceleration == pytest.approx(12.566370614359172)
        for limit in joint_limits
    )
