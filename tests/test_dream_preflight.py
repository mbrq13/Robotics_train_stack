from __future__ import annotations

import json
from pathlib import Path

import yaml

from robotics_stack.config.loaders import load_yaml
from robotics_stack.evaluation.dream_preflight import check_dream_deployment


def test_offline_dream_preflight_accepts_the_committed_contract(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    source = root / "configs" / "deploy" / "dream1_piper.yaml"
    deployment = load_yaml(source)
    deployment["arm_file"] = str(root / "assets" / "robots" / "piper" / "piper_arm.json")
    # Physical confirmation is deliberately absent from the repository. This
    # test exercises every static contract around that operator-only gate.
    deployment["require_tcp_confirmation"] = False
    test_config = tmp_path / "dream.yaml"
    test_config.write_text(yaml.safe_dump(deployment), encoding="utf-8")

    report = check_dream_deployment(
        root / "configs" / "robots" / "piper_bimanual_dream1.yaml",
        test_config,
        check_api=False,
        check_devices=False,
    )

    assert report.robot == "bipiper-dream1-v1"
    assert report.cameras == {"left": (640, 360), "top": (640, 360), "right": (640, 360)}
    assert report.arm["tcp_confirmed"] is False


def test_offline_preflight_accepts_one_physical_so100_arm(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    robot = load_yaml(root / "configs" / "robots" / "so100_single_dream1.yaml")
    calibration = tmp_path / "so100.json"
    names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    calibration.write_text(
        json.dumps(
            {
                name: {
                    "id": index,
                    "drive_mode": 0,
                    "homing_offset": 0,
                    "range_min": 1000,
                    "range_max": 3000,
                }
                for index, name in enumerate(names, start=1)
            }
        ),
        encoding="utf-8",
    )
    robot["robot"]["options"].update(
        {
            "port": "/dev/serial/by-id/test-so100",
            "calibration_file": str(calibration),
            "home": [0.0, -0.8, 0.7, 1.1, 0.0],
        }
    )
    robot_config = tmp_path / "robot.yaml"
    robot_config.write_text(yaml.safe_dump(robot), encoding="utf-8")

    report = check_dream_deployment(
        robot_config,
        root / "configs" / "deploy" / "dream1_so100_single.yaml",
        check_api=False,
        check_devices=False,
    )

    assert report.robot == "so100-single-dream1-v1"
    assert report.devices["arm_serial"] == "/dev/serial/by-id/test-so100"
    assert report.arm == {
        "ik_arm": "so101_faas",
        "physical_arms": 1,
        "commanded_side": "right",
        "calibration_file": str(calibration),
    }
