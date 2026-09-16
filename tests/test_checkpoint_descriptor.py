from __future__ import annotations

import json

import pytest

from robotics_stack.contracts import ContractError, JointLimit, RobotSchema
from robotics_stack.policy.checkpoints import inspect_checkpoint


def _station() -> RobotSchema:
    names = tuple(
        [*(f"left_joint_{index}" for index in range(1, 7)), "left_gripper"]
        + [*(f"right_joint_{index}" for index in range(1, 7)), "right_gripper"]
    )
    return RobotSchema(
        name="bipiper",
        action_names=names,
        state_names=names,
        joint_limits=tuple(JointLimit(-3, 3, 1, 1) for _ in names),
        camera_names=("left", "top", "right"),
    )


def test_pi05_descriptor_requires_order_and_cameras(tmp_path) -> None:
    names = [
        *(f"left_joint{index}" for index in range(1, 7)),
        "left_gripper",
        *(f"right_joint{index}" for index in range(1, 7)),
        "right_gripper",
    ]
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {
                    "observation.state": {"shape": [14]},
                    "observation.images.left": {"shape": [3, 480, 640]},
                    "observation.images.top": {"shape": [3, 376, 672]},
                    "observation.images.right": {"shape": [3, 480, 640]},
                },
                "output_features": {"action": {"shape": [14]}},
                "state_feature_names": names,
                "action_feature_names": names,
            }
        ),
        encoding="utf-8",
    )
    descriptor = inspect_checkpoint(tmp_path, state_names=tuple(names))
    assert descriptor.kind == "pi05"
    assert descriptor.cameras["top"] == (3, 376, 672)
    descriptor.validate_station(_station())

    wrong = _station()
    object.__setattr__(wrong, "camera_names", ("left", "right"))
    with pytest.raises(ContractError, match="missing cameras"):
        descriptor.validate_station(wrong)


def test_pi05_requires_explicit_state_order_when_not_in_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {"observation.state": {"shape": [14]}},
                "output_features": {"action": {"shape": [14]}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="state order"):
        inspect_checkpoint(tmp_path).validate_station(_station())
