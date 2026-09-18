"""Headset adapters that normalize tracking into a device-independent frame."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ControllerSample:
    """Pose and controls for one hand in an OpenXR-style local space."""

    pose: tuple[float, float, float, float, float, float, float]
    tracked: bool
    primary_pressed: bool = False
    secondary_pressed: bool = False
    stick_y: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(self.pose, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("controller pose must contain seven finite values")


@dataclass(frozen=True)
class TrackingFrame:
    """One synchronized pair of controller samples in a named source frame."""

    timestamp_ns: int
    left: ControllerSample
    right: ControllerSample
    source_space: str = "local"


class TrackingProvider(Protocol):
    """Pull interface that allows device and test providers to be exchanged."""

    def poll(self) -> TrackingFrame | None: ...


def _vector(value: Any, *, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, dict):
        keys = ("x", "y", "z", "w")[:size]
        value = tuple(value.get(key, default[index]) for index, key in enumerate(keys))
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return default
    return result if len(result) == size and np.isfinite(result).all() else default


def _controller(payload: dict[str, Any], side: str) -> ControllerSample:
    prefix = "left" if side == "left" else "right"
    position = _vector(payload.get(f"{prefix}ControllerPosition"), size=3, default=(0.0, 0.0, 0.0))
    orientation = _vector(
        payload.get(f"{prefix}ControllerRotation"), size=4, default=(0.0, 0.0, 0.0, 1.0)
    )
    stick = _vector(payload.get(f"{prefix}Joystick"), size=2, default=(0.0, 0.0))
    tracked = bool(payload.get(f"{prefix}Tracked", False))
    valid = bool(payload.get(f"{prefix}Valid", True))
    primary_key = "buttonXPressed" if side == "left" else "buttonAPressed"
    secondary_key = "buttonYPressed" if side == "left" else "buttonBPressed"
    return ControllerSample(
        pose=position + orientation,
        tracked=tracked and valid,
        primary_pressed=bool(payload.get(primary_key, False)),
        secondary_pressed=bool(payload.get(secondary_key, False)),
        stick_y=stick[1],
    )


def parse_quest_frame(payload: dict[str, Any], *, received_ns: int | None = None) -> TrackingFrame:
    """Parse the documented newline-delimited Quest controller payload."""
    return TrackingFrame(
        timestamp_ns=int(received_ns if received_ns is not None else time.monotonic_ns()),
        left=_controller(payload, "left"),
        right=_controller(payload, "right"),
        source_space=str(payload.get("space", "local")),
    )


class QuestTrackingClient:
    """Read newline-delimited controller frames from a Quest companion app."""

    def __init__(self, host: str, *, port: int = 65432, reconnect_s: float = 1.0) -> None:
        if not host or not 0 < port < 65536 or reconnect_s <= 0.0:
            raise ValueError("Quest connection settings are invalid")
        self.host = host
        self.port = port
        self.reconnect_s = reconnect_s
        self._latest: TrackingFrame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="quest-tracking")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def poll(self) -> TrackingFrame | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=2.0) as connection:
                    stream = connection.makefile("r", encoding="utf-8")
                    for line in stream:
                        if self._stop.is_set():
                            return
                        try:
                            frame = parse_quest_frame(
                                json.loads(line), received_ns=time.monotonic_ns()
                            )
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            self.last_error = f"invalid Quest frame: {exc}"
                            continue
                        with self._lock:
                            self._latest = frame
            except OSError as exc:
                self.last_error = str(exc)
                self._stop.wait(self.reconnect_s)


class PicoTrackingProvider:
    """Optional adapter for a locally installed XRoboToolkit Python SDK."""

    tracking_port = 63901

    def __init__(self, sdk: Any | None = None) -> None:
        if sdk is None:
            try:
                import xrobotoolkit_sdk as sdk_module
            except ImportError as exc:
                raise RuntimeError(
                    "install the PICO XRoboToolkit SDK to use PICO tracking"
                ) from exc
            sdk = sdk_module
            sdk.init()
        self.sdk = sdk

    @classmethod
    def prepare_usb(cls) -> None:
        """Set up the PICO USB reverse tunnel without changing CAN state."""
        commands = (
            ("adb", "reverse", f"tcp:{cls.tracking_port}", f"tcp:{cls.tracking_port}"),
            ("adb", "shell", "svc", "power", "stayon", "usb"),
        )
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=10.0)

    def poll(self) -> TrackingFrame | None:
        try:
            left = self._sample("left")
            right = self._sample("right")
        except Exception:
            return None
        return TrackingFrame(timestamp_ns=time.monotonic_ns(), left=left, right=right)

    def _sample(self, side: str) -> ControllerSample:
        pose_reader = getattr(self.sdk, f"get_{side}_controller_pose")
        pose = _vector(pose_reader(), size=7, default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        button = "X" if side == "left" else "A"
        primary = bool(float(getattr(self.sdk, f"get_{button}_button")()))
        stick_reader = getattr(self.sdk, f"get_{side}_joystick", None)
        stick = _vector(stick_reader() if stick_reader else (0.0, 0.0), size=2, default=(0.0, 0.0))
        return ControllerSample(pose=pose, tracked=True, primary_pressed=primary, stick_y=stick[1])
