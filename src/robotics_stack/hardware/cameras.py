"""Bounded, latest-frame camera capture for the robot station."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


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
    crop: str = "none"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height and fps must be positive")
        if self.source_width is not None and self.source_width < self.width:
            raise ValueError("source_width cannot be smaller than output width")
        if self.crop not in {"none", "left_half"}:
            raise ValueError("camera crop must be 'none' or 'left_half'")
        if self.crop == "left_half" and self.source_width != self.width * 2:
            raise ValueError("left_half crop requires source_width to equal twice width")


@dataclass(frozen=True)
class CameraHealth:
    connected: bool
    age_ms: float | None
    error: str | None
    shape: tuple[int, int] | None


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
            capture = cv2.VideoCapture(config.device)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.source_width or config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
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

    def frames(self) -> dict[str, bytes]:
        with self._lock:
            return dict(self._frames)

    def frame(self, name: str) -> bytes:
        """Return the latest already-encoded preview without touching capture."""
        with self._lock:
            try:
                return self._frames[name]
            except KeyError as exc:
                raise KeyError(f"camera {name!r} has no frame") from exc

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
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok:
                with self._lock:
                    self._errors[config.name] = "frame capture failed"
                self._stop.wait(0.05)
                continue
            if config.crop == "left_half":
                frame = frame[:, : frame.shape[1] // 2]
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
