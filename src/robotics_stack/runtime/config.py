"""Small, explicit YAML configuration loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robotics_stack.contracts import RobotSchema
from robotics_stack.hardware.cameras import CameraConfig
from robotics_stack.hardware.piper import PiperConnection


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def load_station_config(path: str | Path) -> tuple[RobotSchema, dict[str, Any], PiperConnection]:
    value = load_yaml(path)
    schema = RobotSchema.from_dict(value["schema"])
    station = value.get("station", {})
    piper = PiperConnection(**value.get("piper", {}))
    return schema, station, piper


def load_camera_configs(value: dict[str, Any]) -> list[CameraConfig]:
    return [CameraConfig(**camera) for camera in value.get("cameras", [])]
