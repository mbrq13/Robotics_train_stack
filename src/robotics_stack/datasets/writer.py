"""Durable native-LeRobot episode recording with bounded background workers."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robotics_stack.contracts import RobotSchema
from robotics_stack.datasets.capture import (
    CaptureCoordinator,
    CapturedFrame,
    CaptureRequest,
)


class DatasetWriteError(RuntimeError):
    """Capture or persistent storage could not keep a dataset row valid."""


class LeRobotDatasetLike(Protocol):
    """Small surface used from the public LeRobot dataset API."""

    num_episodes: int
    num_frames: int
    features: dict[str, Any]

    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def clear_episode_buffer(self) -> None: ...

    def finalize(self) -> None: ...


@dataclass(frozen=True)
class LeRobotRecordingConfig:
    """Dataset identity and bounded worker capacity for one recording session."""

    root: Path
    repo_id: str
    task: str
    camera_shapes: dict[str, tuple[int, int]]
    fps: float = 30.0
    resume: bool = False
    capture_queue_size: int = 2
    writer_queue_size: int = 60
    durable_episodes: bool = True
    video_codec: str = "h264"

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("repo_id must not be empty")
        if self.fps <= 0 or self.capture_queue_size <= 0 or self.writer_queue_size <= 0:
            raise ValueError("recording rates and queue sizes must be positive")
        if int(self.fps) != self.fps:
            raise ValueError("native LeRobot recording requires an integer fps")
        if not self.video_codec.strip():
            raise ValueError("video codec must not be empty")
        if any(height <= 0 or width <= 0 for height, width in self.camera_shapes.values()):
            raise ValueError("camera shapes must be positive (height, width) pairs")


@dataclass
class _Request:
    operation: str
    payload: Any = None
    done: threading.Event | None = None
    error: BaseException | None = None


def _recording_features(
    schema: RobotSchema, camera_shapes: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    action_vector = {
        "dtype": "float32",
        "shape": (schema.action_size,),
        "names": list(schema.action_names),
    }
    state_vector = {**action_vector, "names": list(schema.state_names)}
    features: dict[str, Any] = {
        "observation.state": state_vector,
        "action": action_vector,
        "recording.source": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["policy=0,operator=1"],
        },
        "recording.control_generation": {"dtype": "int64", "shape": (1,), "names": None},
        "recording.command_timestamp_ns": {"dtype": "int64", "shape": (1,), "names": None},
        "recording.state_timestamp_ns": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for name, (height, width) in camera_shapes.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
        features[f"recording.camera_timestamp_ns.{name}"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
    return features


def open_lerobot_dataset(
    config: LeRobotRecordingConfig, schema: RobotSchema
) -> LeRobotDatasetLike:
    """Open a native LeRobot dataset without importing it for non-recording commands."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("install the policy-lerobot extra to record LeRobot episodes") from exc
    kwargs = {
        "repo_id": config.repo_id,
        "root": config.root,
        "streaming_encoding": bool(config.camera_shapes),
        "encoder_queue_maxsize": max(
            round(config.fps), round(config.fps) * len(config.camera_shapes)
        ),
        "vcodec": config.video_codec,
    }
    if config.resume:
        dataset = LeRobotDataset.resume(**kwargs)
        expected = set(_recording_features(schema, config.camera_shapes))
        actual = set(dataset.features)
        if missing := expected - actual:
            raise DatasetWriteError(
                "existing dataset is missing required recording features: "
                f"{', '.join(sorted(missing))}"
            )
        return dataset
    if config.root.exists() and any(config.root.iterdir()):
        raise DatasetWriteError("dataset root is not empty; use --resume or a new --output-dir")
    return LeRobotDataset.create(
        **kwargs,
        fps=int(config.fps),
        robot_type=schema.robot_type or schema.name,
        features=_recording_features(schema, config.camera_shapes),
        use_videos=bool(config.camera_shapes),
    )


class LeRobotEpisodeRecorder:
    """Capture station-local rows and commit native episodes without control-loop I/O.

    There are two bounded queues. The first protects command-to-sensor timing;
    the second protects the encoder and filesystem. Either saturation becomes a
    visible failure, never a silent dropped frame.
    """

    def __init__(
        self,
        *,
        schema: RobotSchema,
        coordinator: CaptureCoordinator,
        config: LeRobotRecordingConfig,
        dataset_factory: Callable[
            [LeRobotRecordingConfig, RobotSchema], LeRobotDatasetLike
        ] = open_lerobot_dataset,
    ) -> None:
        self.schema = schema
        self.coordinator = coordinator
        self.config = config
        self.dataset = dataset_factory(config, schema)
        self._capture_queue: queue.Queue[_Request] = queue.Queue(config.capture_queue_size)
        self._writer_queue: queue.Queue[_Request] = queue.Queue(config.writer_queue_size)
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._closed = False
        self._episode_open = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="dataset-capture"
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="dataset-writer"
        )
        self._capture_thread.start()
        self._writer_thread.start()

    @property
    def episode_open(self) -> bool:
        return self._episode_open

    @property
    def frame_count(self) -> int:
        return int(getattr(self, "_frame_count", 0))

    def submit(self, request: CaptureRequest) -> bool:
        """Queue an accepted station action when the configured dataset tick is due."""
        self.raise_if_failed()
        if self._closed or not self._episode_open:
            raise DatasetWriteError("no active dataset episode")
        if not self.coordinator.due(request.emitted_monotonic_ns):
            return False
        self._enqueue(self._capture_queue, _Request("capture", request), "capture")
        return True

    def save_episode(self) -> int:
        """Flush both workers, commit one episode, validate it, and begin the next."""
        self.raise_if_failed()
        frames = self._synchronize("save")
        self._episode_open = True
        return int(frames)

    def discard_episode(self) -> int:
        """Flush capture work then atomically clear the unfinished episode buffer."""
        if self._closed:
            return 0
        frames = self._synchronize("discard", allow_failed=True)
        self._episode_open = True
        return int(frames or 0)

    def close(self, *, save_active: bool = False) -> int:
        """Close worker threads. An unfinished episode is discarded by default."""
        if self._closed:
            return 0
        saved = 0
        try:
            if self._episode_open:
                saved = (
                    self.save_episode()
                    if save_active and self.frame_count
                    else self.discard_episode()
                )
        finally:
            self._closed = True
            self._stop(self._capture_queue, self._capture_thread)
            self._stop(self._writer_queue, self._writer_thread)
            try:
                self.dataset.finalize()
            except Exception:
                if self._failure is None:
                    raise
        return saved

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise DatasetWriteError(
                f"dataset recording failed: {type(failure).__name__}: {failure}"
            ) from failure

    def _synchronize(self, operation: str, *, allow_failed: bool = False) -> int | None:
        capture = _Request("barrier", done=threading.Event())
        self._enqueue(self._capture_queue, capture, "capture")
        capture.done.wait()
        if capture.error is not None and not allow_failed:
            raise DatasetWriteError("capture worker failed") from capture.error
        writer = _Request(operation, done=threading.Event())
        self._enqueue(self._writer_queue, writer, "writer")
        writer.done.wait()
        if writer.error is not None and not allow_failed:
            raise DatasetWriteError(f"could not {operation} dataset episode") from writer.error
        if not allow_failed:
            self.raise_if_failed()
        return writer.payload

    def _enqueue(self, target: queue.Queue[_Request], request: _Request, label: str) -> None:
        try:
            target.put_nowait(request)
        except queue.Full as exc:
            error = DatasetWriteError(f"{label} queue is full; recording cannot remain aligned")
            self._fail(error)
            raise error from exc

    def _capture_loop(self) -> None:
        while True:
            request = self._capture_queue.get()
            try:
                if request.operation == "stop":
                    return
                if request.operation == "capture":
                    try:
                        frame = self.coordinator.capture(request.payload)
                        self._enqueue(self._writer_queue, _Request("frame", frame), "writer")
                    except BaseException as exc:
                        self._fail(exc)
                elif request.operation != "barrier":
                    raise RuntimeError(f"unknown capture request {request.operation!r}")
            except BaseException as exc:
                request.error = exc
                self._fail(exc)
            finally:
                if request.done is not None:
                    request.done.set()
                self._capture_queue.task_done()

    def _writer_loop(self) -> None:
        self._frame_count = 0
        while True:
            request = self._writer_queue.get()
            try:
                if request.operation == "stop":
                    return
                if request.operation == "frame":
                    self.dataset.add_frame(self._to_lerobot_frame(request.payload))
                    self._frame_count += 1
                elif request.operation == "save":
                    request.payload = self._frame_count
                    if self._frame_count:
                        self.dataset.save_episode()
                        if self.config.durable_episodes:
                            self._durably_reopen()
                    self._frame_count = 0
                elif request.operation == "discard":
                    request.payload = self._frame_count
                    self.dataset.clear_episode_buffer()
                    self._frame_count = 0
                else:
                    raise RuntimeError(f"unknown writer request {request.operation!r}")
            except BaseException as exc:
                request.error = exc
                self._fail(exc)
            finally:
                if request.done is not None:
                    request.done.set()
                self._writer_queue.task_done()

    def _to_lerobot_frame(self, frame: CapturedFrame) -> dict[str, Any]:
        output: dict[str, Any] = {
            "observation.state": np.asarray(frame.state, dtype=np.float32),
            "action": np.asarray(frame.request.action, dtype=np.float32),
            "task": self.config.task,
            "recording.source": np.asarray([frame.request.source], dtype=np.int64),
            "recording.control_generation": np.asarray(
                [frame.request.control_generation], dtype=np.int64
            ),
            "recording.command_timestamp_ns": np.asarray(
                [frame.request.emitted_monotonic_ns], dtype=np.int64
            ),
            "recording.state_timestamp_ns": np.asarray([frame.state_monotonic_ns], dtype=np.int64),
        }
        for name, camera in frame.cameras.items():
            output[f"observation.images.{name}"] = self._decode_camera(name, camera.data)
            output[f"recording.camera_timestamp_ns.{name}"] = np.asarray(
                [camera.monotonic_ns], dtype=np.int64
            )
        return output

    def _decode_camera(self, name: str, encoded: bytes) -> np.ndarray:
        """Decode station JPEG outside control/capture workers for video encoding."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise DatasetWriteError("install Pillow to record camera video") from exc
        image = np.asarray(Image.open(BytesIO(encoded)).convert("RGB"))
        expected = self.config.camera_shapes[name]
        if image.shape != (*expected, 3):
            raise DatasetWriteError(
                f"camera {name!r} decoded as {image.shape}; expected {(*expected, 3)}"
            )
        return image

    def _durably_reopen(self) -> None:
        """Close shard writers per episode, check metadata, then append safely."""
        expected_episode = int(self.dataset.num_episodes)
        self.dataset.finalize()
        info_path = self.config.root / "meta" / "info.json"
        if not info_path.is_file():
            raise DatasetWriteError("LeRobot finalize did not produce meta/info.json")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if int(info.get("total_episodes", -1)) != expected_episode:
            raise DatasetWriteError("finalized dataset episode count does not match the writer")
        reopened = LeRobotRecordingConfig(**{**self.config.__dict__, "resume": True})
        self.dataset = open_lerobot_dataset(reopened, self.schema)

    def _fail(self, exc: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = exc

    @staticmethod
    def _stop(target: queue.Queue[_Request], thread: threading.Thread) -> None:
        request = _Request("stop", done=threading.Event())
        target.put(request)
        request.done.wait(timeout=2.0)
        thread.join(timeout=2.0)
