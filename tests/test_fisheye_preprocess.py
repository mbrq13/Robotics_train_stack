from __future__ import annotations

import numpy as np

from robotics_stack.processors.fisheye import (
    FisheyeRectification,
    approximate_rectification_maps,
    center_crop_16_9,
    load_camera_preprocess_profile,
    rectify_equidistant_fisheye,
)


def test_center_crop_preserves_16_9_without_squashing() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[60:420] = 255
    output = center_crop_16_9(frame)
    assert output.shape == (360, 640, 3)
    assert output.min() == 255


def test_approximate_fisheye_map_is_centered_and_valid() -> None:
    config = FisheyeRectification()
    map_x, map_y, metrics = approximate_rectification_maps((640, 480), config)
    assert map_x.shape == (360, 640)
    assert map_y.shape == (360, 640)
    assert np.isclose(map_x[179:181, 319:321].mean(), 319.5, atol=0.01)
    assert np.isclose(map_y[179:181, 319:321].mean(), 239.5, atol=0.01)
    assert metrics["valid_output_fraction"] == 1.0
    assert 250 < metrics["mapped_source_width_px"] < 300
    assert 160 < metrics["mapped_source_height_px"] < 200


def test_rectified_output_has_policy_shape() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output, metrics = rectify_equidistant_fisheye(frame, FisheyeRectification())
    assert output.shape == (360, 640, 3)
    assert 53 < metrics["target_vfov_deg"] < 55


def test_svpro_profile_is_camera_owned_and_targets_only_wrist_views() -> None:
    profile = load_camera_preprocess_profile("configs/cameras/svpro_imx335_fisheye.yaml")
    assert profile.kind == "equidistant_fisheye"
    assert profile.roles == ("left", "right")
    assert profile.rectification.target_hfov_deg == 84.0
