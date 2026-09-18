from __future__ import annotations

from pathlib import Path

import pytest

from robotics_stack.contracts import JointLimit, RobotSchema
from robotics_stack.datasets.capture import (
    CaptureCoordinator,
    CaptureError,
    CaptureRequest,
    CaptureSettings,
)
from robotics_stack.datasets.writer import (
    DatasetWriteError,
    LeRobotEpisodeRecorder,
    LeRobotRecordingConfig,
)
from robotics_stack.robots.cameras import CameraFrame


class FakeDataset:
    def __init__(self) -> None:
        self.features: dict[str, object] = {}
        self.num_episodes = 0
        self.num_frames = 0
        self.pending: list[dict] = []
        self.episodes: list[list[dict]] = []
        self.finalized = False

    def add_frame(self, frame: dict) -> None:
        self.pending.append(frame)

    def save_episode(self) -> None:
        self.episodes.append(self.pending)
        self.num_episodes += 1
        self.num_frames += len(self.pending)
        self.pending = []

    def clear_episode_buffer(self) -> None:
        self.pending = []

    def finalize(self) -> None:
        self.finalized = True


def _schema() -> RobotSchema:
    return RobotSchema(
        name="test",
        robot_type="test_robot",
        action_names=("joint_1", "joint_2"),
        state_names=("joint_1", "joint_2"),
        joint_limits=(JointLimit(-1, 1, 1, 1), JointLimit(-1, 1, 1, 1)),
        camera_names=("front",),
    )


def _coordinator(*, max_age_ms: float = 100.0) -> CaptureCoordinator:
    clock_values = iter((1_000_000_001, 1_000_000_002, 1_000_000_003))
    return CaptureCoordinator(
        settings=CaptureSettings(fps=30, max_capture_delay_ms=100, max_camera_age_ms=max_age_ms),
        action_size=2,
        camera_names=("front",),
        read_state=lambda: (0.1, -0.1),
        read_cameras=lambda: {
            "front": CameraFrame(b"jpeg", 1_000_000_000, (240, 320)),
        },
        clock_ns=lambda: next(clock_values),
    )


def _state_coordinator() -> CaptureCoordinator:
    clock_values = iter((1_000_000_001, 1_000_000_002))
    return CaptureCoordinator(
        settings=CaptureSettings(fps=30),
        action_size=2,
        camera_names=(),
        read_state=lambda: (0.1, -0.1),
        read_cameras=lambda: {},
        clock_ns=lambda: next(clock_values),
    )


def test_capture_reserves_rate_before_background_work() -> None:
    coordinator = _coordinator()

    assert coordinator.due(1_000_000_000)
    assert not coordinator.due(1_010_000_000)
    assert coordinator.due(1_034_000_000)


def test_capture_rejects_stale_camera_frame() -> None:
    coordinator = _coordinator(max_age_ms=0.000001)
    request = CaptureRequest((0.2, -0.2), 1_000_000_000, source=1, control_generation=3)

    with pytest.raises(CaptureError, match="camera frame is too old"):
        coordinator.capture(request)


def test_native_recorder_writes_station_action_and_temporal_provenance(tmp_path: Path) -> None:
    dataset = FakeDataset()
    recorder = LeRobotEpisodeRecorder(
        schema=_schema(),
        coordinator=_state_coordinator(),
        config=LeRobotRecordingConfig(
            root=tmp_path / "dataset",
            repo_id="local/test",
            task="place",
            camera_shapes={},
            durable_episodes=False,
        ),
        dataset_factory=lambda _config, _schema: dataset,
    )
    try:
        assert recorder.submit(CaptureRequest((0.2, -0.2), 1_000_000_000, 1, 3))
        assert recorder.save_episode() == 1
    finally:
        recorder.close()

    frame = dataset.episodes[0][0]
    assert frame["task"] == "place"
    assert frame["action"].tolist() == pytest.approx([0.2, -0.2])
    assert frame["recording.source"].tolist() == [1]
    assert frame["recording.control_generation"].tolist() == [3]
    assert dataset.finalized


def test_writer_failure_becomes_a_visible_recording_fault(tmp_path: Path) -> None:
    class FailingDataset(FakeDataset):
        def add_frame(self, frame: dict) -> None:
            del frame
            raise OSError("disk unavailable")

    recorder = LeRobotEpisodeRecorder(
        schema=_schema(),
        coordinator=_state_coordinator(),
        config=LeRobotRecordingConfig(
            root=tmp_path / "dataset",
            repo_id="local/test",
            task="place",
            camera_shapes={},
            durable_episodes=False,
        ),
        dataset_factory=lambda _config, _schema: FailingDataset(),
    )
    try:
        recorder.submit(CaptureRequest((0.2, -0.2), 1_000_000_000, 1, 3))
        with pytest.raises(DatasetWriteError, match="recording failed"):
            recorder.save_episode()
    finally:
        recorder.close()
