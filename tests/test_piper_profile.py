from pathlib import Path

import pytest

from robotics_stack.config.loaders import load_station_config


def test_default_piper_profile_uses_physical_checkpoint_coordinates() -> None:
    root = Path(__file__).parents[1]
    config = root / "configs" / "robots" / "piper_bimanual.yaml"
    schema, station, _connection = load_station_config(config)

    assert station["motion_mode"] == "direct"
    joint_limits = [limit for index, limit in enumerate(schema.joint_limits) if index % 7 != 6]
    assert joint_limits[0].minimum == pytest.approx(-2.617993878)
    assert joint_limits[1].maximum == pytest.approx(3.141592654)
