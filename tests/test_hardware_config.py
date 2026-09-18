from __future__ import annotations

import pytest

from robotics_stack.robots.cameras import CameraConfig
from robotics_stack.robots.piper import _read_gripper_calibration


def test_stereo_camera_contract_requires_the_real_source_shape() -> None:
    config = CameraConfig(
        name="top",
        device="/dev/video0",
        width=672,
        height=376,
        source_width=1344,
        crop="left_half",
    )
    assert config.source_width == 1344
    with pytest.raises(ValueError, match="twice width"):
        CameraConfig(
            name="top",
            device="/dev/video0",
            width=672,
            height=376,
            source_width=672,
            crop="left_half",
        )


def test_gripper_calibration_is_independent_per_arm(tmp_path) -> None:
    path = tmp_path / "grippers.yaml"
    path.write_text(
        "left:\n  closed_microm: -100\n  open_microm: 70100\n"
        "right:\n  closed_microm: 50\n  open_microm: 69900\n",
        encoding="utf-8",
    )
    assert _read_gripper_calibration(str(path), "left") == (-100, 70100)
    assert _read_gripper_calibration(str(path), "right") == (50, 69900)
