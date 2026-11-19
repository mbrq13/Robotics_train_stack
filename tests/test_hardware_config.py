from __future__ import annotations

import numpy as np
import pytest

from robotics_stack.config.loaders import load_camera_configs, load_yaml
from robotics_stack.robots.cameras import CameraConfig, _crop_frame
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


def test_dream_camera_contract_preserves_source_and_policy_shapes() -> None:
    wrist = CameraConfig(
        name="left",
        device="/dev/video0",
        source_width=640,
        source_height=480,
        width=640,
        height=360,
        crop="center_16_9",
    )
    top = CameraConfig(
        name="top",
        device="/dev/video1",
        source_width=1344,
        source_height=376,
        width=640,
        height=360,
        crop="left_half_center_16_9",
        strict_source_shape=True,
    )
    assert (wrist.source_width, wrist.source_height) == (640, 480)
    assert (top.source_width, top.source_height) == (1344, 376)
    assert top.strict_source_shape is True


def test_dream_crops_wrist_and_zed_left_eye_without_aspect_distortion() -> None:
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    stereo = np.zeros((376, 1344, 3), dtype=np.uint8)
    stereo[:, 672:, :] = 255

    wrist_crop = _crop_frame(wrist, "center_16_9")
    top_crop = _crop_frame(stereo, "left_half_center_16_9")

    assert wrist_crop.shape == (360, 640, 3)
    assert top_crop.shape == (376, 668, 3)
    assert not top_crop.any(), "the right ZED eye must not enter the policy image"


def test_gripper_calibration_is_independent_per_arm(tmp_path) -> None:
    path = tmp_path / "grippers.yaml"
    path.write_text(
        "left:\n  closed_microm: -100\n  open_microm: 70100\n"
        "right:\n  closed_microm: 50\n  open_microm: 69900\n",
        encoding="utf-8",
    )
    assert _read_gripper_calibration(str(path), "left") == (-100, 70100)
    assert _read_gripper_calibration(str(path), "right") == (50, 69900)


def test_uncalibrated_wrist_cameras_use_center_crop_only() -> None:
    value = load_yaml("configs/robots/piper_bimanual_dream1.yaml")
    cameras = {camera.name: camera for camera in load_camera_configs(value)}
    assert cameras["left"].preprocess_profile is None
    assert cameras["right"].preprocess_profile is None
    assert cameras["top"].preprocess_profile is None
    assert cameras["left"].crop == cameras["right"].crop == "center_16_9"
