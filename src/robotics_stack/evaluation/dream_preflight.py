"""Fail-closed preflight for a Dream-1 Piper deployment."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotics_stack.config.loaders import load_camera_configs, load_station_config, load_yaml
from robotics_stack.contracts import ContractError
from robotics_stack.policies.dream1 import (
    Dream1Config,
    dream_api_preflight,
    validate_arm_file,
    validate_dream_schema,
)


def _check_can_interface(port: str) -> None:
    try:
        result = subprocess.run(
            ["ip", "-details", "-json", "link", "show", port],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        link = json.loads(result.stdout)[0]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
        raise ContractError(f"cannot inspect CAN interface {port}: {exc}") from exc
    if "UP" not in link.get("flags", []):
        raise ContractError(f"CAN interface {port} is not UP")
    bitrate = link.get("linkinfo", {}).get("info_data", {}).get("bitrate")
    if bitrate != 1_000_000:
        raise ContractError(f"CAN interface {port} bitrate is {bitrate}, expected 1000000")


@dataclass(frozen=True)
class DreamPreflight:
    robot: str
    cameras: dict[str, tuple[int, int]]
    devices: dict[str, str]
    arm: dict[str, Any]
    api: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "cameras": self.cameras,
            "devices": self.devices,
            "arm": self.arm,
            "api": self.api,
        }


def check_dream_deployment(
    robot_config: str | Path,
    dream_config: str | Path,
    *,
    check_api: bool = True,
    check_devices: bool = False,
) -> DreamPreflight:
    schema, station, profile = load_station_config(robot_config)
    dream = Dream1Config.load(dream_config)
    validate_dream_schema(schema, dream)
    serial_port: str | None = None
    calibration_file: Path | None = None
    if dream.mode == "so100_single":
        if profile.driver != "so100_single":
            raise ContractError("Dream-1 SO-100 deployment requires robot.driver=so100_single")
        serial_port = str(profile.options.get("port", ""))
        if not serial_port.startswith(("/dev/serial/by-id/", "/dev/serial/by-path/")):
            raise ContractError("SO-100 port must be a stable /dev/serial/by-id or by-path name")
        raw_home = profile.options.get("home", ())
        try:
            home = tuple(float(value) for value in raw_home)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "replace every SO-100 home placeholder with measured radians"
            ) from exc
        if len(home) != 5:
            raise ContractError("SO-100 home must contain five measured radians")
        schema.validate_action([*home, 1.0])
        calibration_file = Path(
            str(profile.options.get("calibration_file", ""))
        ).expanduser()
        if not calibration_file.is_file():
            raise ContractError(f"SO-100 calibration file does not exist: {calibration_file}")
        try:
            from robots import so101

            so101.gripper_ends(str(calibration_file))
        except ImportError as exc:
            raise ContractError("install the so100 dependency before preflight") from exc
        except Exception as exc:
            raise ContractError(f"invalid SO-100 calibration file: {exc}") from exc
    else:
        if profile.driver != "piper_bimanual":
            raise ContractError("Dream-1 Piper deployment requires robot.driver=piper_bimanual")
        left_port = str(profile.options.get("left_port", ""))
        right_port = str(profile.options.get("right_port", ""))
        if not left_port or not right_port or left_port == right_port:
            raise ContractError("left_port and right_port must name two distinct CAN interfaces")

    if str(station.get("motion_mode")) != "bounded":
        raise ContractError("Dream-1 hardware deployment requires station.motion_mode=bounded")
    minimum_first_action_ms = 1_000 * dream.first_chunk_timeout_s
    if float(station.get("first_action_timeout_ms", 0)) < minimum_first_action_ms:
        raise ContractError(
            "first_action_timeout_ms is shorter than Dream first_chunk_timeout_s"
        )
    minimum_chunk_age_ms = 1_000 * (dream.action_timeout_s + 48 / dream.action_hz)
    if float(station.get("scheduled_action_age_ms", 0)) < minimum_chunk_age_ms:
        raise ContractError(
            "scheduled_action_age_ms cannot cover inference timeout plus one Dream chunk"
        )

    cameras = load_camera_configs(load_yaml(robot_config))
    by_name = {camera.name: camera for camera in cameras}
    missing = sorted(set(dream.camera_roles) - set(by_name))
    if missing:
        raise ContractError(f"robot config is missing Dream cameras: {', '.join(missing)}")
    shapes = {name: (by_name[name].width, by_name[name].height) for name in dream.camera_roles}
    wrong_shape = {name: shape for name, shape in shapes.items() if shape != (640, 360)}
    if wrong_shape:
        raise ContractError(f"Dream policy cameras must be 640x360, got {wrong_shape}")
    devices = {name: str(by_name[name].device) for name in dream.camera_roles}
    if dream.mode == "so100_single":
        assert serial_port is not None
        devices["arm_serial"] = serial_port
    else:
        devices.update({"can_left": left_port, "can_right": right_port})
    camera_devices = [devices[name] for name in dream.camera_roles]
    if len(set(camera_devices)) != len(camera_devices):
        raise ContractError("Dream cameras must use three distinct device paths")
    if check_devices:
        absent = [
            device
            for device in camera_devices
            if not device.isdigit() and not Path(device).exists()
        ]
        if absent:
            raise ContractError(f"camera device paths do not exist: {', '.join(absent)}")
        resolved = [str(Path(device).resolve()) for device in camera_devices]
        if len(set(resolved)) != len(resolved):
            raise ContractError("two Dream camera paths resolve to the same video device")
        if dream.mode == "so100_single":
            assert serial_port is not None and calibration_file is not None
            if not Path(serial_port).exists():
                raise ContractError(f"SO-100 serial port does not exist: {serial_port}")
        else:
            absent_can = [
                port
                for port in (left_port, right_port)
                if not Path("/sys/class/net", port).exists()
            ]
            if absent_can:
                raise ContractError(f"CAN interfaces do not exist: {', '.join(absent_can)}")
            for port in (left_port, right_port):
                _check_can_interface(port)

    arm = (
        {
            "ik_arm": dream.ik_arm,
            "physical_arms": 1,
            "commanded_side": "right",
            "calibration_file": str(calibration_file),
        }
        if dream.mode == "so100_single"
        else validate_arm_file(
            dream.arm_file, require_confirmation=dream.require_tcp_confirmation
        )
    )
    api = dream_api_preflight(dream) if check_api else None
    return DreamPreflight(schema.name, shapes, devices, arm, api)
