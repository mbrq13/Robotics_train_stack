"""Static deployment checks that run before hardware is armed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from robotics_stack.config.loaders import load_camera_configs, load_station_config, load_yaml
from robotics_stack.contracts import ContractError
from robotics_stack.policies.checkpoints import CheckpointDescriptor, inspect_checkpoint
from robotics_stack.rollout.rtc import RtcSettings


@dataclass(frozen=True)
class DeploymentPreflight:
    checkpoint: CheckpointDescriptor
    execution_mode: str
    action_space: str
    camera_shapes: dict[str, tuple[int, int, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": asdict(self.checkpoint),
            "execution_mode": self.execution_mode,
            "action_space": self.action_space,
            "camera_shapes": self.camera_shapes,
        }


def check_deployment(
    station_config: str | Path,
    worker_config: str | Path,
    checkpoint: str | Path,
) -> DeploymentPreflight:
    """Reject mismatched checkpoint, camera and RTC settings without loading weights."""
    schema, station, profile = load_station_config(station_config)
    worker = load_yaml(worker_config)
    expected_action_space = str(worker.get("action_space", ""))
    expected_robot_type = str(worker.get("robot_type", ""))
    if expected_action_space and profile.action_space != expected_action_space:
        raise ContractError(
            "worker action_space does not match station hardware profile: "
            f"worker={expected_action_space!r}, station={profile.action_space!r}"
        )
    if schema.action_space and schema.action_space != profile.action_space:
        raise ContractError(
            "station schema action_space does not match hardware profile: "
            f"schema={schema.action_space!r}, hardware={profile.action_space!r}"
        )
    if expected_robot_type and schema.robot_type != expected_robot_type:
        raise ContractError(
            "worker robot_type does not match station schema: "
            f"worker={expected_robot_type!r}, station={schema.robot_type!r}"
        )
    state_names = tuple(str(name) for name in worker.get("state_names", ()))
    descriptor = inspect_checkpoint(checkpoint, state_names=state_names)
    descriptor.validate_station(schema)

    cameras = load_camera_configs(load_yaml(station_config))
    camera_shapes = {camera.name: (3, camera.height, camera.width) for camera in cameras}
    for name, expected in descriptor.cameras.items():
        if expected and camera_shapes.get(name) != expected:
            actual = camera_shapes.get(name)
            raise ContractError(f"camera {name} shape station={actual}, checkpoint={expected}")

    execution_mode = str(worker.get("execution_mode", "sync"))
    if execution_mode == "standard":
        execution_mode = "sync"
    if execution_mode not in {"sync", "rtc"}:
        raise ValueError("execution_mode must be sync or rtc")
    action_smoothing_alpha = float(worker.get("action_smoothing_alpha", 1.0))
    if not 0.0 < action_smoothing_alpha <= 1.0:
        raise ContractError("action_smoothing_alpha must be in (0, 1]")
    if execution_mode == "rtc" and action_smoothing_alpha != 1.0:
        raise ContractError(
            "action smoothing is not supported with RTC because it changes queued actions "
            "after the trained prefix is constructed"
        )
    if execution_mode == "rtc":
        delay = descriptor.rtc_training_max_delay
        if delay <= 0:
            raise ContractError("RTC requires a checkpoint trained with rtc_training_max_delay")
        horizon = int(worker.get("rtc_execution_horizon", delay))
        threshold = int(worker.get("rtc_refill_threshold", max(delay, horizon)))
        settings = RtcSettings(
            control_hz=schema.control_hz,
            training_max_delay=delay,
            execution_horizon=horizon,
            refill_threshold=threshold,
        )
        settings.validate_chunk_size(descriptor.chunk_size)
        scheduled_age_ms = float(
            station.get("scheduled_action_age_ms", station.get("action_age_ms", 250))
        )
        minimum_scheduled_age_ms = 1_000 * (delay + horizon) / schema.control_hz
        if scheduled_age_ms < minimum_scheduled_age_ms:
            raise ContractError(
                "scheduled_action_age_ms is shorter than the RTC queue lifetime "
                f"({scheduled_age_ms:.1f} ms < {minimum_scheduled_age_ms:.1f} ms)"
            )

    return DeploymentPreflight(descriptor, execution_mode, profile.action_space, camera_shapes)
