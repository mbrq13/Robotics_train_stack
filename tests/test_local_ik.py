from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robotics_stack.contracts import ContractError, Observation
from robotics_stack.kinematics.dls import DampedLeastSquares
from robotics_stack.kinematics.transforms import (
    anchored_target,
    matrix_pose,
    origin_transform,
    pose_matrix,
    rotation_vector,
)
from robotics_stack.kinematics.urdf import UrdfArm
from robotics_stack.policies.dream1 import Dream1Config, Dream1Policy
from robotics_stack.policies.dream_ik import DreamDls

ROOT = Path(__file__).parents[1]
ARM = ROOT / "assets/robots/piper/piper_arm.json"


def config():
    return Dream1Config(
        station_url="ws://localhost:8765",
        checkpoint="dreamlabs/dream-1-beta",
        task="test",
        arm_file=ARM,
        camera_roles={"left": "wrist_l", "right": "wrist_r", "top": "overhead"},
        ik_solve="dls",
        require_tcp_confirmation=False,
    )


def test_urdf_jacobian_matches_finite_difference_and_tool_axes():
    arm = UrdfArm(ARM)
    q = np.array([0.1, 0.8, -0.7, 0.2, 0.3, -0.2])
    pose = arm.forward(q)
    jac = arm.jacobian(q)
    for i in range(6):
        moved = q.copy()
        moved[i] += 1e-6
        other = arm.forward(moved)
        assert np.allclose(jac[:3, i], (other[:3, 3] - pose[:3, 3]) / 1e-6, atol=1e-5)
        assert np.allclose(
            jac[3:, i], rotation_vector(other[:3, :3] @ pose[:3, :3].T) / 1e-6, atol=1e-5
        )
    assert np.allclose(arm.tool[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


def reply_for(arm, left, right, rows=8):
    parts = {}
    for side, q in (("left", left), ("right", right)):
        pose = arm.forward(q)
        parts[f"action.{side}_eef_xyz"] = np.tile(pose[:3, 3], (rows, 1))
        parts[f"action.{side}_eef_rot6"] = np.tile(np.r_[pose[:3, 0], pose[:3, 1]], (rows, 1))
        parts[f"action.{side}_grip"] = np.full((rows, 1), 0.2 if side == "left" else 0.8)
    return {"per_key": parts}


def test_dls_absolute_targets_bounds_and_side_order():
    adapter = DreamDls(config())
    q = np.array([0.0, 0.8, -0.7, 0.0, 0.4, 0.0])
    goal = q.copy()
    goal[1] += 0.08
    state = np.r_[q, 0.0, q, 1.0]
    chunk = adapter.solve(reply_for(adapter.arm, goal, q), state)
    assert chunk.shape == (8, 14)
    assert np.all(chunk[:, 6] == 0.2) and np.all(chunk[:, 13] == 0.8)
    sequence = np.vstack([state, chunk])
    assert np.max(np.abs(np.diff(sequence[:, :6], axis=0))) <= 0.5 / 20 + 1e-10
    assert np.all(chunk[:, :6] >= adapter.arm.lower)
    assert np.all(chunk[:, :6] <= adapter.arm.upper)
    before = np.linalg.norm(adapter.arm.forward(q)[:3, 3] - adapter.arm.forward(goal)[:3, 3])
    after = np.linalg.norm(
        adapter.arm.forward(chunk[-1, :6])[:3, 3] - adapter.arm.forward(goal)[:3, 3]
    )
    assert after < before
    bad = reply_for(adapter.arm, goal, q)
    bad["per_key"]["action.left_eef_rot6"][:] = 0
    with pytest.raises(ContractError, match="Degenerate"):
        adapter.solve(bad, state)


def test_workspace_and_keepout_are_enforced():
    q = np.array([0.0, 0.8, -0.7, 0.0, 0.4, 0.0])
    base = DreamDls(config())
    reply = reply_for(base.arm, q, q)
    limited = DreamDls(
        replace(config(), ik_region={"left": {"lo": [10, 10, 10], "hi": [11, 11, 11]}})
    )
    with pytest.raises(ContractError, match="workspace"):
        limited.solve(reply, np.r_[q, 0, q, 0])
    blocked = DreamDls(
        replace(
            config(),
            ik_keep_out={
                "left": [{"kind": "capsule", "center": [0, 0, 0], "height": 1, "radius": 1}]
            },
        )
    )
    with pytest.raises(ContractError, match="keep-out"):
        blocked.solve(reply, np.r_[q, 0, q, 0])


def test_policy_uses_raw_eef_without_sdk_ik():
    class Client:
        def policy(self, value):
            self.configuration = value

        def set_observation(self, frames, payload, **kwargs):
            self.payload = payload

        def get_action(self, **kwargs):
            return self.reply

    client = Client()
    policy = Dream1Policy(config(), client=client)
    q = np.array([0.0, 0.8, -0.7, 0.0, 0.4, 0.0])
    state = tuple(np.r_[q, 0, q, 1])
    policy.set_observation(
        Observation(1, 1_000_000, state, {"left": b"a", "right": b"b", "top": b"c"})
    )
    assert "ik" not in client.configuration
    assert np.all(client.payload["state.left_arm"] == 0)
    assert np.allclose(client.payload["state.left_eef_xyz"], policy._local_ik.arm.forward(q)[:3, 3])
    client.reply = reply_for(policy._local_ik.arm, q, q)
    assert policy.get_chunk().shape == (8, 14)


def test_rotation_log_at_pi():
    assert np.linalg.norm(rotation_vector(np.diag([1.0, -1.0, -1.0]))) == pytest.approx(np.pi)


@pytest.mark.parametrize("rate", [float("inf"), float("nan"), 0.0])
def test_dls_rejects_invalid_rate(rate):
    arm = UrdfArm(ARM)
    with pytest.raises(ValueError, match="nominal rate"):
        DampedLeastSquares(
            arm,
            rest_joints=arm.rest,
            lower=arm.lower,
            upper=arm.upper,
            max_speed_rad_s=arm.speed,
            nominal_rate_hz=rate,
        )


def test_dream_config_rejects_unused_keys(tmp_path):
    path = tmp_path / "dream.yaml"
    path.write_text("ik_solver: dls\n")
    with pytest.raises(ValueError, match="Unknown Dream configuration keys: ik_solver"):
        Dream1Config.load(path)


def test_anchored_delta_maps_world_axes_with_rotated_anchors():
    source = origin_transform((0.2, -0.1, 0.3), (0.4, -0.3, 1.2))
    robot = origin_transform((0.3, 0.1, 0.4), (-0.7, 0.6, -0.5))
    world_map = origin_transform((0, 0, 0), (0.2, 0.1, -0.8))[:3, :3]
    rotation_delta = origin_transform((0, 0, 0), (0.05, -0.07, 0.1))[:3, :3]
    current = source.copy()
    current[:3, 3] += [0.01, -0.02, 0.03]
    current[:3, :3] = rotation_delta @ source[:3, :3]
    target = pose_matrix(
        anchored_target(
            matrix_pose(source),
            matrix_pose(current),
            matrix_pose(robot),
            source_to_robot_rotation=world_map,
            translation_scale=1.5,
            max_displacement_m=0.6,
        )
    )
    assert np.allclose(
        target[:3, 3], robot[:3, 3] + world_map @ np.array([0.01, -0.02, 0.03]) * 1.5
    )
    assert np.allclose(target[:3, :3], world_map @ rotation_delta @ world_map.T @ robot[:3, :3])
    # Check the equivalent composition in the two anchored local frames.
    adapter = source[:3, :3].T @ world_map.T @ robot[:3, :3]
    local_delta = np.linalg.inv(source) @ current
    expected = np.eye(4)
    expected[:3, :3] = adapter.T @ local_delta[:3, :3] @ adapter
    expected[:3, 3] = adapter.T @ local_delta[:3, 3] * 1.5
    assert np.allclose(target, robot @ expected)


def test_anchored_target_holds_on_engagement_and_caps_displacement():
    anchor = matrix_pose(origin_transform((0.1, 0.2, 0.3), (0.3, -0.4, 0.8)))
    robot = matrix_pose(origin_transform((0.3, 0.0, 0.4), (-0.2, 0.5, 0.1)))
    kwargs = dict(source_to_robot_rotation=np.eye(3), translation_scale=1.0, max_displacement_m=0.1)
    assert np.allclose(
        pose_matrix(anchored_target(anchor, anchor, robot, **kwargs)), pose_matrix(robot)
    )
    current = anchor.copy()
    current[0] += 1
    target = anchored_target(anchor, current, robot, **kwargs)
    assert np.allclose(target[:3] - robot[:3], [0.1, 0, 0])
