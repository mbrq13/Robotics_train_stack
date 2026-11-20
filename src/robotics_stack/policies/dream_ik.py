"""Local incremental IK adapter for absolute Dream tool-pose chunks."""

from __future__ import annotations

import numpy as np

from robotics_stack.contracts import ContractError
from robotics_stack.kinematics.dls import DampedLeastSquares, DlsSettings
from robotics_stack.kinematics.transforms import matrix_pose
from robotics_stack.kinematics.urdf import UrdfArm


class DreamDls:
    def __init__(self, config):
        self.config = config
        self.arm = UrdfArm(config.arm_file)
        if len(self.arm.lower) != 6:
            raise ValueError("Piper DLS requires six revolute joints")
        self.solver = DampedLeastSquares(
            self.arm,
            rest_joints=self.arm.rest,
            lower=self.arm.lower,
            upper=self.arm.upper,
            max_speed_rad_s=np.minimum(self.arm.speed, 0.5),
            nominal_rate_hz=config.action_hz,
            settings=DlsSettings(**(config.dls or {})),
        )
        for obstacles in (config.ik_keep_out or {}).values():
            for obstacle in obstacles:
                if obstacle.get("kind") != "capsule":
                    raise ValueError("DLS keep-out supports only vertical capsules")
                if float(obstacle["radius"]) <= 0 or float(obstacle["height"]) <= 0:
                    raise ValueError("Keep-out dimensions must be positive")

    def observation(self, state, payload):
        for side, start in (("left", 0), ("right", 7)):
            pose = self.arm.forward(state[start : start + 6])
            payload[f"state.{side}_eef_xyz"] = pose[:3, 3]
            payload[f"state.{side}_eef_rot6"] = np.concatenate((pose[:3, 0], pose[:3, 1]))
            # Base checkpoint uses its training robot's joint slots as placeholders.
            payload[f"state.{side}_arm"] = np.zeros(6)
            payload[f"state.{side}_gripper"] = state[start + 6 : start + 7]
        return payload

    def _check_workspace(self, side, joints, target):
        region = (self.config.ik_region or {}).get(side)
        points = self.arm.link_positions(joints)
        if region:
            lo, hi = np.asarray(region["lo"]), np.asarray(region["hi"])
            if np.any(target < lo) or np.any(target > hi):
                raise ContractError(f"{side} DLS tool target is outside workspace")
            if np.any(points < lo) or np.any(points > hi):
                raise ContractError(f"{side} DLS links are outside workspace")
        # Check sampled link segments with a 4 cm link radius. These static guards
        # do not model inter-arm collisions or replace a measured scene model.
        samples = np.concatenate(
            [
                np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a) / 0.01)) + 1))
                for a, b in zip(points[:-1], points[1:], strict=True)
            ]
        )
        for obstacle in (self.config.ik_keep_out or {}).get(side, []):
            center = np.asarray(obstacle["center"], dtype=float)
            nearest = np.tile(center, (len(samples), 1))
            nearest[:, 2] = np.clip(
                samples[:, 2],
                center[2] - obstacle["height"] / 2,
                center[2] + obstacle["height"] / 2,
            )
            if np.any(np.linalg.norm(samples - nearest, axis=1) < obstacle["radius"] + 0.045):
                raise ContractError(f"{side} DLS links intersect keep-out")

    def solve(self, reply, state):
        parts = reply.get("per_key", {})
        if not isinstance(parts, dict):
            raise ContractError("DLS requires named EEF action parts")
        chunks = []
        count = None
        for side, start in (("left", 0), ("right", 7)):
            try:
                xyz = np.asarray(parts[f"action.{side}_eef_xyz"], dtype=float)
                rot = np.asarray(parts[f"action.{side}_eef_rot6"], dtype=float)
                grip = np.asarray(parts[f"action.{side}_grip"], dtype=float)
            except (KeyError, ValueError, TypeError) as exc:
                raise ContractError("DLS requires named EEF action parts") from exc
            if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
                raise ContractError("Invalid EEF positions")
            n = len(xyz)
            if grip.shape == (n,):
                grip = grip[:, None]
            if rot.shape != (n, 6) or grip.shape != (n, 1) or count not in (None, n):
                raise ContractError("Inconsistent EEF chunk shapes")
            count = n
            if not all(np.isfinite(v).all() for v in (xyz, rot, grip)):
                raise ContractError("Non-finite EEF chunk")
            q = np.asarray(state[start : start + 6], dtype=float).copy()
            rows = []
            for position, orientation, opening in zip(xyz, rot, grip, strict=True):
                x = orientation[:3].copy()
                if np.linalg.norm(x) < 1e-8:
                    raise ContractError("Degenerate rot6 orientation")
                x /= np.linalg.norm(x)
                y = orientation[3:] - np.dot(x, orientation[3:]) * x
                if np.linalg.norm(y) < 1e-8:
                    raise ContractError("Degenerate rot6 orientation")
                y /= np.linalg.norm(y)
                pose = np.eye(4)
                pose[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
                pose[:3, 3] = position
                candidate = self.solver.step(q, matrix_pose(pose), dt_s=1 / self.config.action_hz)
                self._check_workspace(side, candidate, position)
                q = candidate
                rows.append(np.r_[q, np.clip(opening[0], 0, 1)])
            chunks.append(np.asarray(rows))
        return np.concatenate(chunks, axis=1)
