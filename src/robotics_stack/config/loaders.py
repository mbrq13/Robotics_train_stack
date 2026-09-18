"""Small, explicit YAML configuration loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robotics_stack.contracts import RobotSchema
from robotics_stack.robots.cameras import CameraConfig
from robotics_stack.robots.registry import RobotProfile


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def load_station_config(path: str | Path) -> tuple[RobotSchema, dict[str, Any], RobotProfile]:
    value = load_yaml(path)
    schema = RobotSchema.from_dict(value["schema"])
    station = value.get("station", {})
    profile = RobotProfile.from_dict(value["robot"])
    if schema.action_space != profile.action_space:
        raise ValueError("schema.action_space must match robot.action_space")
    return schema, station, profile


def load_camera_configs(value: dict[str, Any]) -> list[CameraConfig]:
    return [CameraConfig(**camera) for camera in value.get("cameras", [])]
