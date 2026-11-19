"""Bounded, latest-frame camera capture for the robot station."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotics_stack.processors.fisheye import (
    approximate_rectification_maps,
    load_camera_preprocess_profile,
)


def _crop_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    """Apply geometry-only crops before the final policy resize."""
    if mode.startswith("left_half"):
        frame = frame[:, : frame.shape[1] // 2]
    if mode.endswith("center_16_9"):
        target_ratio = 16.0 / 9.0
        ratio = frame.shape[1] / frame.shape[0]
        if ratio > target_ratio:
            target_width = int(round(frame.shape[0] * target_ratio))
            offset = (frame.shape[1] - target_width) // 2
            frame = frame[:, offset : offset + target_width]
        elif ratio < target_ratio:
            target_height = int(round(frame.shape[1] / target_ratio))
            offset = (frame.shape[0] - target_height) // 2
            frame = frame[offset : offset + target_height, :]
    return frame


@dataclass(frozen=True)
class CameraConfig:
    name: str
    device: str | int
    width: int
    height: int
    fps: float = 30.0
    jpeg_quality: int = 80
    # Some stereo UVC cameras publish both eyes side by side. The model sees
    # the cropped output, while OpenCV must request the wider source frame.
    source_width: int | None = None
    source_height: int | None = None
    crop: str = "none"
    backend: str = "V4L2"
    fourcc: str | None = "MJPG"
    fisheye_calibration_file: str | None = None
    preprocess_profile: str | None = None
    strict_source_shape: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height and fps must be positive")
        if self.source_width is not None and self.source_width <= 0:
            raise ValueError("source_width must be positive")
        if self.source_height is not None and self.source_height <= 0:
            raise ValueError("source_height must be positive")
        crops = {"none", "left_half", "center_16_9", "left_half_center_16_9"}
        if self.crop not in crops:
            raise ValueError(f"camera crop must be one of {sorted(crops)}")
        if self.crop.startswith("left_half"):
            if self.source_width is None or self.source_width % 2:
                raise ValueError("left-half crop requires an even source_width")
            if self.crop == "left_half" and self.source_width // 2 != self.width:
                raise ValueError("left_half crop requires source_width to equal twice width")
        if self.fourcc is not None and len(self.fourcc) != 4:
            raise ValueError("camera fourcc must contain exactly four characters")
        if self.backend.upper() not in {"ANY", "V4L2"}:
            raise ValueError("camera backend must be ANY or V4L2")
        if self.fisheye_calibration_file is not None and self.preprocess_profile is not None:
            raise ValueError(
                "camera cannot use fisheye_calibration_file and preprocess_profile together"
            )


@dataclass(frozen=True)
class CameraHealth:
    connected: bool
    age_ms: float | None
    error: str | None
    shape: tuple[int, int] | None


@dataclass(frozen=True)
class CameraFrame:
    """One encoded camera frame together with its station-clock capture time."""

    data: bytes
    monotonic_ns: int
    shape: tuple[int, int]


class CameraHub:
    """Captures only the newest frame; slow consumers never create a backlog."""

    def __init__(self, cameras: list[CameraConfig]):
        self.cameras = cameras
        self._captures: dict[str, Any] = {}
        self._frames: dict[str, bytes] = {}
        self._timestamps: dict[str, int] = {}
        self._shapes: dict[str, tuple[int, int]] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def connect(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("install opencv to use cameras") from exc
        self._stop.clear()
        for config in self.cameras:
            backend = cv2.CAP_V4L2 if config.backend.upper() == "V4L2" else cv2.CAP_ANY
            capture = cv2.VideoCapture(config.device, backend)
            if config.fourcc:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*config.fourcc))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.source_width or config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.source_height or config.height)
            capture.set(cv2.CAP_PROP_FPS, config.fps)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"camera {config.name} cannot open {config.device!r}")
            self._captures[config.name] = capture
            thread = threading.Thread(
                target=self._capture_loop,
                args=(config,),
                daemon=True,
                name=f"camera-{config.name}",
            )
            thread.start()
            self._threads.append(thread)

    def disconnect(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def frames(
        self,
        *,
        max_age_ms: float | None = None,
        max_skew_ms: float | None = None,
    ) -> dict[str, bytes]:
        with self._lock:
            if len(self._frames) != len(self.cameras):
                missing = sorted({camera.name for camera in self.cameras} - set(self._frames))
                raise RuntimeError(f"camera frames are not ready: {', '.join(missing)}")
            now = time.monotonic_ns()
            if max_age_ms is not None:
                stale = [
                    name
                    for name, timestamp in self._timestamps.items()
                    if (now - timestamp) / 1_000_000 > max_age_ms
                ]
                if stale:
                    raise RuntimeError(f"camera frames are stale: {', '.join(sorted(stale))}")
            if max_skew_ms is not None and self._timestamps:
                skew_ms = (
                    max(self._timestamps.values()) - min(self._timestamps.values())
                ) / 1_000_000
                if skew_ms > max_skew_ms:
                    raise RuntimeError(
                        f"camera frame skew {skew_ms:.1f} ms exceeds {max_skew_ms:.1f} ms"
                    )
            return dict(self._frames)

    def frame(self, name: str) -> bytes:
        """Return the latest already-encoded preview without touching capture."""
        return self.snapshot(name).data

    def snapshot(self, name: str) -> CameraFrame:
        """Return the latest frame and its timestamp as one atomic snapshot."""
        with self._lock:
            try:
                return CameraFrame(
                    data=self._frames[name],
                    monotonic_ns=self._timestamps[name],
                    shape=self._shapes[name],
                )
            except KeyError as exc:
                raise KeyError(f"camera {name!r} has no frame") from exc

    def snapshots(self) -> dict[str, CameraFrame]:
        """Return a consistent copy of the newest frame from every camera."""
        with self._lock:
            return {
                name: CameraFrame(self._frames[name], self._timestamps[name], self._shapes[name])
                for name in self._frames
            }

    def health(self) -> dict[str, CameraHealth]:
        now = time.monotonic_ns()
        with self._lock:
            return {
                config.name: CameraHealth(
                    connected=config.name in self._frames,
                    age_ms=(now - self._timestamps[config.name]) / 1_000_000
                    if config.name in self._timestamps
                    else None,
                    error=self._errors.get(config.name),
                    shape=self._shapes.get(config.name),
                )
                for config in self.cameras
            }

    def validate(self, max_age_ms: float = 500.0) -> None:
        health_by_name = self.health()
        failed = [
            config.name
            for config in self.cameras
            if (
                not (health := health_by_name[config.name]).connected
                or health.age_ms is None
                or health.age_ms > max_age_ms
                or health.shape != (config.height, config.width)
            )
        ]
        if failed:
            raise RuntimeError(f"camera validation failed: {', '.join(failed)}")

    def _capture_loop(self, config: CameraConfig) -> None:
        import cv2

        capture = self._captures[config.name]
        rectify_maps: tuple[np.ndarray, np.ndarray] | None = None
        if config.fisheye_calibration_file is not None:
            calibration_path = Path(config.fisheye_calibration_file).expanduser()
            try:
                with np.load(calibration_path) as calibration:
                    intrinsic = np.asarray(calibration["K"], dtype=np.float64)
                    distortion = np.asarray(calibration["D"], dtype=np.float64)
                source_size = (
                    int(config.source_width or config.width),
                    int(config.source_height or config.height),
                )
                rectify_maps = cv2.fisheye.initUndistortRectifyMap(
                    intrinsic,
                    distortion,
                    np.eye(3),
                    intrinsic,
                    source_size,
                    cv2.CV_16SC2,
                )
            except (OSError, KeyError, ValueError) as exc:
                with self._lock:
                    self._errors[config.name] = f"invalid fisheye calibration: {exc}"
                return
        elif config.preprocess_profile is not None:
            try:
                profile = load_camera_preprocess_profile(config.preprocess_profile)
                if config.name not in profile.roles:
                    raise ValueError(
                        f"profile roles {profile.roles} do not include camera {config.name!r}"
                    )
                expected_output = (config.width, config.height)
                actual_output = (
                    profile.rectification.output_width,
                    profile.rectification.output_height,
                )
                if actual_output != expected_output:
                    raise ValueError(
                        f"profile output {actual_output[0]}x{actual_output[1]} does not match "
                        f"camera output {expected_output[0]}x{expected_output[1]}"
                    )
                source_size = (
                    int(config.source_width or config.width),
                    int(config.source_height or config.height),
                )
                map_x, map_y, _metrics = approximate_rectification_maps(
                    source_size, profile.rectification
                )
                rectify_maps = (map_x, map_y)
            except (OSError, ValueError) as exc:
                with self._lock:
                    self._errors[config.name] = f"invalid camera preprocess profile: {exc}"
                return
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok:
                with self._lock:
                    self._errors[config.name] = "frame capture failed"
                self._stop.wait(0.05)
                continue
            expected_source = (
                int(config.source_height or config.height),
                int(config.source_width or config.width),
            )
            if config.strict_source_shape and frame.shape[:2] != expected_source:
                with self._lock:
                    self._errors[config.name] = (
                        f"expected source {expected_source[1]}x{expected_source[0]}, "
                        f"got {frame.shape[1]}x{frame.shape[0]}"
                    )
                self._stop.wait(0.05)
                continue
            if rectify_maps is not None:
                frame = cv2.remap(
                    frame,
                    rectify_maps[0],
                    rectify_maps[1],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            frame = _crop_frame(frame, config.crop)
            if (frame.shape[1], frame.shape[0]) != (config.width, config.height):
                frame = cv2.resize(
                    frame,
                    (config.width, config.height),
                    interpolation=cv2.INTER_AREA,
                )
            shape = (int(frame.shape[0]), int(frame.shape[1]))
            if shape != (config.height, config.width):
                with self._lock:
                    self._errors[config.name] = (
                        f"expected {config.width}x{config.height}, got {shape[1]}x{shape[0]}"
                    )
                    self._shapes[config.name] = shape
                self._stop.wait(0.05)
                continue
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
            )
            if not ok:
                with self._lock:
                    self._errors[config.name] = "JPEG encoding failed"
                continue
            with self._lock:
                self._frames[config.name] = encoded.tobytes()
                self._timestamps[config.name] = time.monotonic_ns()
                self._shapes[config.name] = shape
                self._errors.pop(config.name, None)
