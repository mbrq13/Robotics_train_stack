import numpy as np

from robotics_stack.evaluation.dream_dataset_replay import (
    _interpolate_rows,
    _metrics,
    _policy_frame,
    _query_indices,
)


def test_deployment_crop_matches_live_camera_geometry() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:60] = 10
    frame[60:420] = 20
    frame[420:] = 30
    transformed = _policy_frame(frame, "left", "deployment_crop")
    assert transformed.shape == (360, 640, 3)
    assert np.all(transformed == 20)


def test_letterbox_preserves_full_wrist_height() -> None:
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    transformed = _policy_frame(frame, "left", "full_fov_letterbox")
    assert transformed.shape == (360, 640, 3)
    assert np.all(transformed[:, :80] == 0)
    assert np.all(transformed[:, 80:560] == 100)


def test_metrics_report_degrees_and_exclude_grippers_from_joint_error() -> None:
    reference = np.zeros((2, 14))
    predicted = reference.copy()
    predicted[:, 0] = np.deg2rad(10)
    predicted[:, 6] = 0.5
    metrics = _metrics(predicted, reference, np.zeros(14))
    assert 2.8 < metrics["joint_rmse_deg"] < 2.9
    assert metrics["first_target_delta_deg_max"] == 10
    assert metrics["gripper_mae"] == 0.25


def test_query_indices_have_history_and_cover_episode() -> None:
    assert _query_indices(513, 3, 1.6, 30) == [77, 256, 435]


def test_trajectory_interpolation_preserves_vector_columns() -> None:
    values = np.asarray([[0.0, 10.0], [2.0, 14.0]])
    result = _interpolate_rows(np.asarray([0.0, 2.0]), values, np.asarray([0.5, 1.5]))
    assert result.tolist() == [[0.5, 11.0], [1.5, 13.0]]
