from pathlib import Path

import pytest

from robotics_stack.runtime.config import load_station_config


def test_default_piper_profile_uses_checkpoint_compatible_coordinates() -> None:
    root = Path(__file__).parents[1]
    schema, station, _connection = load_station_config(root / "configs" / "piper_station.yaml")

    assert station["motion_mode"] == "direct"
    joint_limits = [limit for index, limit in enumerate(schema.joint_limits) if index % 7 != 6]
    assert all(limit.minimum == pytest.approx(-100.0) for limit in joint_limits)
    assert all(limit.maximum == pytest.approx(100.0) for limit in joint_limits)
