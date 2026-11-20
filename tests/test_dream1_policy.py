from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from robotics_stack.contracts import ContractError, Observation
from robotics_stack.policies.dream1 import (
    Dream1Config,
    Dream1Policy,
    confirm_arm_tcp,
    validate_arm_file,
)


class FakeDreamClient:
    def __init__(self) -> None:
        self.policy_config = None
        self.frames = None
        self.observation = None
        self.timestamp = None
        self.commanded = None
        self.last_timing = {"rtt_ms": 125.0}

    def policy(self, config):
        self.policy_config = config

    def set_observation(self, frames, observation, *, timestamp):
        self.frames = frames
        self.observation = observation
        self.timestamp = timestamp

    def get_action(self, *, commanded=None):
        self.commanded = commanded
        return {
            "q_l_chunk": [[1, 2, 3, 4, 5, 6, 0.25], [7, 8, 9, 10, 11, 12, 0.5]],
            "q_r_chunk": [[13, 14, 15, 16, 17, 18, 0.75], [19, 20, 21, 22, 23, 24, 1.0]],
        }

    def refill_due(self, executed, remaining):
        return executed >= remaining

    def reset_episode(self):
        pass

    def close(self):
        pass


def _assets() -> tuple[Path, Path]:
    root = Path(__file__).parents[1]
    directory = root / "assets" / "robots" / "piper"
    return directory / "piper_arm.json", directory / "piper_arm.urdf"


def _config(arm_file: Path) -> Dream1Config:
    return Dream1Config(
        station_url="ws://localhost:8765",
        checkpoint="dreamlabs/dream-1",
        task="pick up the block",
        arm_file=arm_file,
        camera_roles={"left": "wrist_l", "top": "overhead", "right": "wrist_r"},
        require_tcp_confirmation=False,
    )


def test_dream_policy_maps_left_first_station_state_and_right_first_api() -> None:
    arm_file, _ = _assets()
    client = FakeDreamClient()
    policy = Dream1Policy(_config(arm_file), client=client)
    policy.set_observation(
        Observation(
            observation_id=1,
            station_monotonic_ns=2_000_000_000,
            state=tuple(float(value) for value in range(14)),
            images={"left": b"left", "top": b"top", "right": b"right"},
        )
    )

    assert np.array_equal(client.observation["state.left_arm"], np.arange(6))
    assert np.array_equal(client.observation["state.right_arm"], np.arange(7, 13))
    assert client.observation["state.left_grip"] == [6]
    assert client.observation["state.right_grip"] == [13]
    assert client.policy_config["embodiment"]["state"]["right_grip"] == [23, 24]
    assert client.policy_config["embodiment"]["action"]["left_grip"] == [31, 32]
    assert set(client.frames) == {"video.wrist_l", "video.overhead", "video.wrist_r"}
    assert client.timestamp == 2.0

    chunk = policy.get_chunk(tuple(float(value) for value in range(14)))
    assert chunk.shape == (2, 14)
    assert chunk[0].tolist() == [1, 2, 3, 4, 5, 6, 0.25, 13, 14, 15, 16, 17, 18, 0.75]
    assert client.commanded["q_l_chunk"].tolist() == [0, 1, 2, 3, 4, 5]
    assert client.commanded["q_r_chunk"].tolist() == [7, 8, 9, 10, 11, 12]


def test_tcp_confirmation_is_bound_to_arm_geometry(tmp_path: Path) -> None:
    source_json, source_urdf = _assets()
    shutil.copytree(source_urdf.parent / "meshes", tmp_path / "meshes")
    arm_file = tmp_path / "arm.json"
    urdf = tmp_path / "arm.urdf"
    arm_file.write_text(source_json.read_text(encoding="utf-8"), encoding="utf-8")
    urdf.write_bytes(source_urdf.read_bytes())
    document = json.loads(arm_file.read_text(encoding="utf-8"))
    document["urdf"] = urdf.name
    document.pop("tcp_calibration", None)
    arm_file.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContractError, match="TCP is not confirmed"):
        validate_arm_file(arm_file, require_confirmation=True)
    confirm_arm_tcp(arm_file)
    assert validate_arm_file(arm_file, require_confirmation=True)["tcp_confirmed"] is True

    changed = json.loads(arm_file.read_text(encoding="utf-8"))
    changed["tcp"][2] += 0.001
    arm_file.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ContractError, match="TCP is not confirmed"):
        validate_arm_file(arm_file, require_confirmation=True)


def test_dream_config_requires_exact_three_camera_roles() -> None:
    arm_file, _ = _assets()
    with pytest.raises(ValueError, match="camera_roles"):
        Dream1Config(
            station_url="ws://localhost",
            checkpoint="dreamlabs/dream-1",
            task="move the block",
            arm_file=arm_file,
            camera_roles={"left": "wrist_l", "top": "overhead"},
        ).validate()


def test_dream_policy_saturates_normalized_grippers() -> None:
    arm_file, _ = _assets()
    client = FakeDreamClient()
    policy = Dream1Policy(_config(arm_file), client=client)
    original = client.get_action

    def outside_gripper_range(*, commanded=None):
        reply = original(commanded=commanded)
        reply["q_l_chunk"][0][-1] = -0.1
        reply["q_r_chunk"][0][-1] = 1.1
        return reply

    client.get_action = outside_gripper_range
    chunk = policy.get_chunk()
    assert chunk[0, 6] == 0.0
    assert chunk[0, 13] == 1.0


def test_dream_policy_exposes_only_the_physical_so100_right_arm() -> None:
    class SingleClient(FakeDreamClient):
        def get_action(self, *, commanded=None):
            self.commanded = commanded
            return {
                "q_r_chunk": [
                    [0.1, -0.2, 0.3, -0.4, 0.5, 1.1],
                    [0.2, -0.1, 0.4, -0.3, 0.6, 0.25],
                ],
                "q_l_chunk": [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]],
            }

    config = Dream1Config(
        station_url="ws://localhost:8765",
        checkpoint="dreamlabs/dream-1-beta",
        task="pick up the block",
        arm_file=None,
        camera_roles={"left": "wrist_l", "top": "overhead", "right": "wrist_r"},
        mode="so100_single",
        ik_arm="so101_faas",
    )
    config.validate()
    client = SingleClient()
    policy = Dream1Policy(config, client=client)
    policy.set_observation(
        Observation(
            observation_id=1,
            station_monotonic_ns=3_000_000_000,
            state=(0.1, -0.2, 0.3, -0.4, 0.5, 0.75),
            images={"left": b"left", "top": b"top", "right": b"right"},
        )
    )

    assert client.policy_config["ik"]["arm"] == "so101_faas"
    assert client.observation["state.right_arm"].tolist() == pytest.approx(
        [0.1, -0.2, 0.3, -0.4, 0.5]
    )
    assert client.observation["state.right_grip"].tolist() == pytest.approx([0.75])
    assert client.observation["state.left_arm"].tolist() == pytest.approx(
        config.dummy_left_joints
    )

    chunk = policy.get_chunk((0.1, -0.2, 0.3, -0.4, 0.5, 0.75))
    assert chunk.shape == (2, 6)
    assert chunk[0].tolist() == pytest.approx([0.1, -0.2, 0.3, -0.4, 0.5, 1.0])
    assert set(client.commanded) == {"q_r_chunk"}
    assert client.commanded["q_r_chunk"].tolist() == pytest.approx(
        [0.1, -0.2, 0.3, -0.4, 0.5]
    )


def test_dream_loads_local_env_without_overriding_exports(tmp_path, monkeypatch):
    from robotics_stack.policies.dream1 import _load_dream_environment

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    (tmp_path / ".env").write_text('DREAM_API_KEY="test-local-key"\n')
    _load_dream_environment()
    import os

    assert os.environ["DREAM_API_KEY"] == "test-local-key"
    monkeypatch.setenv("DREAM_API_KEY", "test-exported-key")
    _load_dream_environment()
    assert os.environ["DREAM_API_KEY"] == "test-exported-key"
