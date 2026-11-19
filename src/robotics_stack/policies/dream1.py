"""Dream-1 zero-shot adapter for supported physical stations.

Dream owns model inference, temporal image windows, chunk pacing and EEF IK.
This module owns the explicit conversion to each station contract and never
imports a hardware driver.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotics_stack.config.loaders import load_yaml
from robotics_stack.contracts import ContractError, Observation, RobotSchema

_PIPER_ACTION_NAMES = (
    *(f"left_joint_{index}" for index in range(1, 7)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(1, 7)),
    "right_gripper",
)
_SO100_ACTION_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_TOOL_KEYS = ("base", "flange", "tcp", "approach", "jaw")


@dataclass(frozen=True)
class Dream1Config:
    station_url: str
    checkpoint: str
    task: str
    arm_file: Path | None
    camera_roles: dict[str, str]
    mode: str = "piper_bimanual"
    ik_arm: str | None = None
    dummy_left_joints: tuple[float, ...] = (
        0.3409,
        -1.7104,
        1.5285,
        1.1724,
        -1.4119,
    )
    dummy_left_grip: float = 1.0
    timeout_ms: int = 15_000
    obs_window: int = 9
    obs_fps: float = 5.0
    action_hz: float = 20.0
    execution_horizon: int = 30
    sync_above_ms: float = 700.0
    action_timeout_s: float = 3.0
    first_chunk_timeout_s: float = 900.0
    require_tcp_confirmation: bool = True
    dry_run: bool = False
    ik_solve: str = "traj"
    ik_region: dict[str, Any] | None = None
    ik_keep_out: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path) -> Dream1Config:
        source = Path(path).resolve()
        value = load_yaml(source)
        arm_value = value.get("arm_file")
        arm_file = None if arm_value is None else Path(str(arm_value)).expanduser()
        if arm_file is not None and not arm_file.is_absolute():
            arm_file = (source.parent / arm_file).resolve()
        roles = {str(name): str(role) for name, role in value.get("camera_roles", {}).items()}
        config = cls(
            station_url=str(value["station_url"]),
            checkpoint=str(value.get("checkpoint", "dreamlabs/dream-1-beta")),
            task=str(value.get("task", "")).strip(),
            arm_file=arm_file,
            camera_roles=roles,
            mode=str(value.get("mode", "piper_bimanual")),
            ik_arm=str(value["ik_arm"]) if value.get("ik_arm") is not None else None,
            dummy_left_joints=tuple(
                float(entry)
                for entry in value.get(
                    "dummy_left_joints",
                    (0.3409, -1.7104, 1.5285, 1.1724, -1.4119),
                )
            ),
            dummy_left_grip=float(value.get("dummy_left_grip", 1.0)),
            timeout_ms=int(value.get("timeout_ms", 15_000)),
            obs_window=int(value.get("obs_window", 9)),
            obs_fps=float(value.get("obs_fps", 5.0)),
            action_hz=float(value.get("action_hz", 20.0)),
            execution_horizon=int(value.get("execution_horizon", 30)),
            sync_above_ms=float(value.get("sync_above_ms", 700.0)),
            action_timeout_s=float(value.get("action_timeout_s", 3.0)),
            first_chunk_timeout_s=float(value.get("first_chunk_timeout_s", 900.0)),
            require_tcp_confirmation=bool(value.get("require_tcp_confirmation", True)),
            dry_run=bool(value.get("dry_run", False)),
            ik_solve=str(value.get("ik_solve", "traj")),
            ik_region=value.get("ik_region"),
            ik_keep_out=value.get("ik_keep_out"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.task:
            raise ValueError("Dream-1 task must not be empty")
        if self.timeout_ms <= 0 or self.obs_window <= 0:
            raise ValueError("Dream-1 timeout_ms and obs_window must be positive")
        if min(
            self.obs_fps,
            self.action_hz,
            self.execution_horizon,
            self.action_timeout_s,
            self.first_chunk_timeout_s,
        ) <= 0:
            raise ValueError("Dream-1 observation and action rates must be positive")
        if self.ik_solve not in {"tick", "traj"}:
            raise ValueError("Dream-1 ik_solve must be tick or traj")
        if self.mode not in {"piper_bimanual", "so100_single"}:
            raise ValueError("Dream-1 mode must be piper_bimanual or so100_single")
        if self.mode == "piper_bimanual" and self.arm_file is None:
            raise ValueError("Dream-1 bimanual Piper mode requires arm_file")
        if self.mode == "so100_single":
            if self.ik_arm != "so101_faas":
                raise ValueError("Dream-1 SO-100 mode requires ik_arm=so101_faas")
            if len(self.dummy_left_joints) != 5 or not np.isfinite(
                self.dummy_left_joints
            ).all():
                raise ValueError("Dream-1 SO-100 dummy_left_joints must be five finite radians")
            if not 0.0 <= self.dummy_left_grip <= 1.0:
                raise ValueError("Dream-1 SO-100 dummy_left_grip must be in [0, 1]")
        required_roles = {"overhead", "wrist_l", "wrist_r"}
        if set(self.camera_roles.values()) != required_roles:
            raise ValueError(
                "camera_roles must map exactly one local camera to overhead, wrist_l and wrist_r"
            )
        if len(self.camera_roles) != len(set(self.camera_roles.values())):
            raise ValueError("Dream-1 camera roles must be one-to-one")


def validate_piper_schema(schema: RobotSchema, action_hz: float) -> None:
    if schema.robot_type != "piper":
        raise ContractError(f"Dream-1 Piper worker cannot drive robot_type={schema.robot_type!r}")
    if schema.action_space != "radians":
        raise ContractError(
            "Dream-1 Piper worker requires physical radians and normalized grippers"
        )
    if schema.action_names != _PIPER_ACTION_NAMES or schema.state_names != _PIPER_ACTION_NAMES:
        raise ContractError("Dream-1 Piper worker requires the canonical left-first 14D schema")
    if not np.isclose(schema.control_hz, action_hz):
        raise ContractError(
            f"station control_hz={schema.control_hz} differs from Dream action_hz={action_hz}"
        )


def validate_dream_schema(schema: RobotSchema, config: Dream1Config) -> None:
    """Match the station to the selected Dream-to-hardware adapter."""
    if config.mode == "piper_bimanual":
        validate_piper_schema(schema, config.action_hz)
        return
    if schema.robot_type != "so100":
        raise ContractError(f"Dream-1 SO-100 worker cannot drive robot_type={schema.robot_type!r}")
    if schema.action_space != "radians":
        raise ContractError("Dream-1 SO-100 worker requires radians and a normalized gripper")
    if schema.action_names != _SO100_ACTION_NAMES or schema.state_names != _SO100_ACTION_NAMES:
        raise ContractError("Dream-1 SO-100 worker requires the canonical single-arm 6D schema")
    if not np.isclose(schema.control_hz, config.action_hz):
        raise ContractError(
            f"station control_hz={schema.control_hz} differs from Dream "
            f"action_hz={config.action_hz}"
        )


def _arm_document(path: Path) -> tuple[dict[str, Any], Path, bytes]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read Dream arm file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Dream arm file {path} must contain an object")
    missing = sorted({"urdf", "base", "flange", "tcp", "approach", "jaw", "rest"} - set(value))
    if missing:
        raise ContractError(f"Dream arm file is missing: {', '.join(missing)}")
    urdf = path.parent / str(value["urdf"])
    try:
        urdf_bytes = urdf.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read Piper URDF {urdf}: {exc}") from exc
    return value, urdf, urdf_bytes


def arm_tool_digest(arm: Mapping[str, Any], urdf: bytes) -> str:
    tool = json.dumps({key: arm[key] for key in _TOOL_KEYS}, sort_keys=True)
    return hashlib.sha256(urdf + b"\0" + tool.encode()).hexdigest()[:16]


def validate_arm_file(path: Path, *, require_confirmation: bool) -> dict[str, Any]:
    arm, urdf, urdf_bytes = _arm_document(path)
    try:
        root = ET.fromstring(urdf_bytes)
    except ET.ParseError as exc:
        raise ContractError(f"invalid Piper URDF: {exc}") from exc
    meshes = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        relative = Path(filename)
        if not filename or "://" in filename or relative.is_absolute():
            raise ContractError("Piper meshes must use portable paths relative to the URDF")
        mesh_path = (urdf.parent / relative).resolve()
        if not mesh_path.is_relative_to(urdf.parent.resolve()):
            raise ContractError("Piper mesh path escapes the asset directory")
        if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
            raise ContractError(f"missing or empty Piper mesh: {filename}")
        meshes.add(filename)
    if len(arm["rest"]) != 6:
        raise ContractError("Piper Dream arm file must declare a six-joint rest pose")
    for key in ("tcp", "approach", "jaw"):
        if not isinstance(arm[key], list) or len(arm[key]) != 3:
            raise ContractError(f"Dream arm file {key} must contain three numbers")
    confirmation = arm.get("tcp_calibration")
    confirmed = (
        isinstance(confirmation, dict)
        and confirmation.get("status") == "calibrated"
        and confirmation.get("digest") == arm_tool_digest(arm, urdf_bytes)
    )
    if require_confirmation and not confirmed:
        raise ContractError(
            "Piper TCP is not confirmed for this URDF; run `rstack dream-confirm-tcp` "
            "after measuring the real gripper"
        )
    return {
        "arm_file": str(path),
        "urdf": str(urdf),
        "tcp": arm["tcp"],
        "tcp_confirmed": confirmed,
        "mesh_count": len(meshes),
    }


def confirm_arm_tcp(path: Path) -> str:
    """Record an operator's explicit confirmation without changing geometry."""
    arm, _urdf, urdf_bytes = _arm_document(path)
    digest = arm_tool_digest(arm, urdf_bytes)
    arm["tcp_calibration"] = {
        "status": "calibrated",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digest": digest,
    }
    path.write_text(json.dumps(arm, indent=2) + "\n", encoding="utf-8")
    return digest


class Dream1Policy:
    """Small stateful facade over the official DreamClient."""

    def __init__(self, config: Dream1Config, *, client: Any | None = None):
        if config.arm_file is not None:
            validate_arm_file(
                config.arm_file, require_confirmation=config.require_tcp_confirmation
            )
        if client is None:
            try:
                import dream
            except ImportError as exc:
                raise RuntimeError("install the dream optional dependency to run Dream-1") from exc
            client = dream.DreamClient()
        self.config = config
        self.client = client
        # Observation updates are intentionally allowed while inference runs: the
        # official client uses them to maintain its temporal window. Lifecycle
        # calls, however, must never reset/close a session during get_action().
        self._request_lock = threading.Lock()
        self._open_policy()

    def _open_policy(self) -> None:
        ik: dict[str, Any] = {
            "arm": (
                str(self.config.arm_file)
                if self.config.arm_file is not None
                else str(self.config.ik_arm)
            ),
            "head": "eef",
            "solve": self.config.ik_solve,
        }
        if self.config.ik_region:
            ik["region"] = self.config.ik_region
        if self.config.ik_keep_out:
            ik["keep_out"] = self.config.ik_keep_out
        video = {
            role: f"observation.images.{local}"
            for local, role in self.config.camera_roles.items()
        }
        embodiment = {
            "state": {
                "right_arm": [0, 6],
                "right_gripper": [6, 7],
                "left_arm": [7, 13],
                "left_gripper": [13, 14],
                "right_eef_xyz": [14, 17],
                "right_eef_rot6": [17, 23],
                "right_grip": [23, 24],
                "left_eef_xyz": [24, 27],
                "left_eef_rot6": [27, 33],
                "left_grip": [33, 34],
            },
            "action": {
                "right_arm": [0, 6],
                "left_arm": [6, 12],
                "right_eef_xyz": [12, 15],
                "right_eef_rot6": [15, 21],
                "right_grip": [21, 22],
                "left_eef_xyz": [22, 25],
                "left_eef_rot6": [25, 31],
                "left_grip": [31, 32],
            },
            "video": video,
        }
        self.client.policy(
            {
                "checkpoint": self.config.checkpoint,
                "timeout_ms": self.config.timeout_ms,
                "strict": True,
                "robot": "other_embodiment",
                "ik": ik,
                "embodiment": embodiment,
                "obs_window": self.config.obs_window,
                "obs_fps": self.config.obs_fps,
                "action_hz": self.config.action_hz,
                "execution_horizon": self.config.execution_horizon,
                "sync_above_ms": self.config.sync_above_ms,
            }
        )

    def set_observation(self, observation: Observation) -> None:
        state = np.asarray(observation.state, dtype=np.float32)
        frames: dict[str, str] = {}
        for local, role in self.config.camera_roles.items():
            try:
                encoded = observation.images[local]
            except KeyError as exc:
                raise ContractError(f"Dream-1 observation is missing camera {local!r}") from exc
            frames[f"video.{role}"] = base64.b64encode(encoded).decode("ascii")
        if self.config.mode == "so100_single":
            if state.shape != (6,):
                raise ContractError("Dream-1 SO-100 requires a single-arm 6D observation")
            payload = {
                "state.right_arm": state[:5],
                "state.right_grip": state[5:6],
                # The checkpoint retains two-hand context.  This fixed virtual
                # arm is an inference anchor only and is never exposed to the
                # station or any hardware driver.
                "state.left_arm": np.asarray(
                    self.config.dummy_left_joints, dtype=np.float32
                ),
                "state.left_grip": np.asarray(
                    [self.config.dummy_left_grip], dtype=np.float32
                ),
                "prompt": self.config.task,
            }
        else:
            if state.shape != (14,):
                raise ContractError("Dream-1 requires a 14D Piper observation")
            payload = {
                "state.right_arm": state[7:13],
                "state.right_grip": state[13:14],
                "state.left_arm": state[0:6],
                "state.left_grip": state[6:7],
                "prompt": self.config.task,
            }
        self.client.set_observation(
            frames,
            payload,
            timestamp=observation.station_monotonic_ns / 1_000_000_000,
        )

    def get_chunk(self, commanded: tuple[float, ...] | None = None) -> np.ndarray:
        seam: dict[str, np.ndarray] | None = None
        if commanded is not None:
            row = np.asarray(commanded, dtype=np.float64)
            if self.config.mode == "so100_single":
                if row.shape != (6,):
                    raise ContractError("Dream-1 SO-100 anchor must be a 6D action")
                seam = {"q_r_chunk": row[:5]}
            else:
                if row.shape != (14,):
                    raise ContractError("Dream-1 commanded anchor must be a 14D Piper action")
                seam = {"q_r_chunk": row[7:13], "q_l_chunk": row[0:6]}
        with self._request_lock:
            reply = self.client.get_action(commanded=seam)
        try:
            right = np.asarray(reply["q_r_chunk"], dtype=np.float64)
            left = np.asarray(reply["q_l_chunk"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("Dream-1 reply has no valid q_l_chunk/q_r_chunk") from exc
        if self.config.mode == "so100_single":
            if right.ndim != 2 or right.shape[0] == 0 or right.shape[1] != 6:
                raise ContractError(
                    f"Dream-1 SO-100 right chunk must have shape [H, 6], got {right.shape}"
                )
            if not np.isfinite(right).all():
                raise ContractError("Dream-1 returned non-finite SO-100 actions")
            right[:, 5] = np.clip(right[:, 5], 0.0, 1.0)
            return right
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ContractError("Dream-1 arm chunks must be equal two-dimensional arrays")
        if left.shape[0] == 0 or left.shape[1] != 7:
            raise ContractError(f"Dream-1 Piper chunks must have shape [H, 7], got {left.shape}")
        chunk = np.concatenate((left, right), axis=1)
        if not np.isfinite(chunk).all():
            raise ContractError("Dream-1 returned non-finite actions")
        # The official physical adapter saturates the normalized gripper drive
        # before it reaches hardware; EEF heads may overshoot slightly at a seam.
        chunk[:, (6, 13)] = np.clip(chunk[:, (6, 13)], 0.0, 1.0)
        return chunk

    def refill_due(self, executed: int, remaining: int) -> bool:
        return bool(self.client.refill_due(executed, remaining))

    def reset_episode(self) -> None:
        with self._request_lock:
            self.client.reset_episode()

    @property
    def timing(self) -> dict[str, Any]:
        return dict(self.client.last_timing or {})

    def close(self) -> None:
        with self._request_lock:
            self.client.close()


def dream_api_preflight(config: Dream1Config) -> dict[str, Any]:
    """Check entitlement and camera roles without opening a policy session."""
    try:
        import dream
    except ImportError as exc:
        raise RuntimeError("install the dream optional dependency to check DreamAPI") from exc
    client = dream.DreamClient()
    try:
        models = client.get_models()
    finally:
        client.close()
    model = next((candidate for candidate in models if candidate.id == config.checkpoint), None)
    if model is None:
        raise ContractError(f"DreamAPI key cannot access checkpoint {config.checkpoint!r}")
    if not model.servable:
        raise ContractError(f"Dream checkpoint {config.checkpoint!r} is not servable")
    expected = sorted(config.camera_roles.values())
    available = sorted(model.cameras or [])
    if available and available != expected:
        raise ContractError(
            f"Dream model cameras are {available}; this deployment supplies {expected}"
        )
    return {
        "checkpoint": model.id,
        "servable": model.servable,
        "model_cameras": available,
    }
