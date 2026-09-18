from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotics_stack.contracts import ContractError
from robotics_stack.evaluation.preflight import check_deployment


def _checkpoint(root: Path, top_shape: list[int]) -> Path:
    root.mkdir()
    names = [
        *(f"left_joint_{index}" for index in range(1, 7)),
        "left_gripper",
        *(f"right_joint_{index}" for index in range(1, 7)),
        "right_gripper",
    ]
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {
                    "observation.state": {"shape": [14]},
                    "observation.images.left": {"shape": [3, 480, 640]},
                    "observation.images.top": {"shape": top_shape},
                    "observation.images.right": {"shape": [3, 480, 640]},
                },
                "output_features": {"action": {"shape": [14]}},
                "state_feature_names": names,
                "chunk_size": 50,
                "rtc_training_max_delay": 10,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_preflight_validates_camera_shape_and_rtc_contract(tmp_path) -> None:
    root = Path(__file__).parents[1]
    worker = tmp_path / "worker.yaml"
    worker.write_text(
        (root / "configs" / "policy_worker.yaml")
        .read_text(encoding="utf-8")
        .replace("execution_mode: sync", "execution_mode: rtc"),
        encoding="utf-8",
    )
    report = check_deployment(
        root / "configs" / "piper_station.yaml",
        worker,
        _checkpoint(tmp_path / "valid", [3, 376, 672]),
    )
    assert report.execution_mode == "rtc"
    assert report.action_space == "radians"
    assert report.camera_shapes["top"] == (3, 376, 672)


def test_preflight_rejects_camera_shape_mismatch(tmp_path) -> None:
    root = Path(__file__).parents[1]
    with pytest.raises(ContractError, match="camera top shape"):
        check_deployment(
            root / "configs" / "piper_station.yaml",
            root / "configs" / "policy_worker.yaml",
            _checkpoint(tmp_path / "invalid", [3, 480, 640]),
        )


def test_preflight_rejects_rtc_queue_that_expires_before_its_last_action(tmp_path) -> None:
    root = Path(__file__).parents[1]
    station = tmp_path / "station.yaml"
    station.write_text(
        (root / "configs" / "piper_station.yaml")
        .read_text(encoding="utf-8")
        .replace("scheduled_action_age_ms: 1000", "scheduled_action_age_ms: 500"),
        encoding="utf-8",
    )
    worker = tmp_path / "worker.yaml"
    worker.write_text(
        (root / "configs" / "policy_worker.yaml")
        .read_text(encoding="utf-8")
        .replace("execution_mode: sync", "execution_mode: rtc"),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="queue lifetime"):
        check_deployment(station, worker, _checkpoint(tmp_path / "valid", [3, 376, 672]))


def test_preflight_rejects_action_space_disagreement(tmp_path) -> None:
    root = Path(__file__).parents[1]
    worker = tmp_path / "worker.yaml"
    worker.write_text(
        (root / "configs" / "policy_worker.yaml")
        .read_text(encoding="utf-8")
        .replace("action_space: radians", "action_space: normalized_100"),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="action_space"):
        check_deployment(
            root / "configs" / "piper_station.yaml",
            worker,
            _checkpoint(tmp_path / "valid", [3, 376, 672]),
        )


def test_preflight_rejects_robot_type_disagreement(tmp_path) -> None:
    root = Path(__file__).parents[1]
    worker = tmp_path / "worker.yaml"
    worker.write_text(
        (root / "configs" / "policy_worker.yaml")
        .read_text(encoding="utf-8")
        .replace("robot_type: piper", "robot_type: another_robot"),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="robot_type"):
        check_deployment(
            root / "configs" / "piper_station.yaml",
            worker,
            _checkpoint(tmp_path / "valid", [3, 376, 672]),
        )


def test_preflight_rejects_smoothing_for_rtc(tmp_path) -> None:
    root = Path(__file__).parents[1]
    worker = tmp_path / "worker.yaml"
    worker.write_text(
        (root / "configs" / "policy_worker.yaml")
        .read_text(encoding="utf-8")
        .replace("execution_mode: sync", "execution_mode: rtc")
        .replace("action_smoothing_alpha: 1.0", "action_smoothing_alpha: 0.5"),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="not supported with RTC"):
        check_deployment(
            root / "configs" / "piper_station.yaml",
            worker,
            _checkpoint(tmp_path / "valid", [3, 376, 672]),
        )
