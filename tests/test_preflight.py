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
        .replace("execution_mode: standard", "execution_mode: rtc"),
        encoding="utf-8",
    )
    report = check_deployment(
        root / "configs" / "piper_station.yaml",
        worker,
        _checkpoint(tmp_path / "valid", [3, 376, 672]),
    )
    assert report.execution_mode == "rtc"
    assert report.camera_shapes["top"] == (3, 376, 672)


def test_preflight_rejects_camera_shape_mismatch(tmp_path) -> None:
    root = Path(__file__).parents[1]
    with pytest.raises(ContractError, match="camera top shape"):
        check_deployment(
            root / "configs" / "piper_station.yaml",
            root / "configs" / "policy_worker.yaml",
            _checkpoint(tmp_path / "invalid", [3, 480, 640]),
        )
