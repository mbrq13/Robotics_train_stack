"""Serial revolute URDF chain with an explicitly defined tool frame."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robotics_stack.kinematics.transforms import origin_transform


class UrdfArm:
    def __init__(self, arm_file: Path):
        spec = json.loads(arm_file.read_text())
        root = ET.parse(arm_file.parent / spec["urdf"]).getroot()
        by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        chain = []
        link = spec["flange"]
        while link != spec["base"]:
            joint = by_child[link]
            if joint in chain:
                raise ValueError("URDF chain contains a cycle")
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain = []
        lower, upper, speed = [], [], []
        for joint in reversed(chain):
            kind = joint.get("type")
            if kind not in {"fixed", "revolute"}:
                raise ValueError("Only fixed and bounded revolute joints are supported")
            origin = joint.find("origin")
            xyz = (
                tuple(map(float, origin.get("xyz", "0 0 0").split()))
                if origin is not None
                else (0, 0, 0)
            )
            rpy = (
                tuple(map(float, origin.get("rpy", "0 0 0").split()))
                if origin is not None
                else (0, 0, 0)
            )
            axis_node = joint.find("axis")
            axis = (
                np.array(list(map(float, axis_node.get("xyz").split())))
                if axis_node is not None
                else np.array([1.0, 0.0, 0.0])
            )
            if np.linalg.norm(axis) == 0:
                raise ValueError("Invalid joint axis")
            axis /= np.linalg.norm(axis)
            self.chain.append((origin_transform(xyz, rpy), axis, kind))
            if kind == "revolute":
                limit = joint.find("limit")
                lower.append(float(limit.get("lower")))
                upper.append(float(limit.get("upper")))
                speed.append(float(limit.get("velocity")))
        self.lower, self.upper, self.speed = map(np.asarray, (lower, upper, speed))
        self.rest = np.asarray(spec["rest"], dtype=float)
        if self.rest.shape != self.lower.shape or not np.isfinite(self.rest).all():
            raise ValueError("Invalid rest posture")
        self.tool = np.eye(4)
        self.tool[:3, 3] = spec["tcp"]
        z = np.asarray(spec["approach"], dtype=float)
        z /= np.linalg.norm(z)
        x = np.asarray(spec["jaw"], dtype=float)
        x -= np.dot(x, z) * z
        x /= np.linalg.norm(x)
        self.tool[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        if not np.isfinite(self.tool).all():
            raise ValueError("Invalid tool frame")

    def _evaluate(self, joints):
        q = np.asarray(joints, dtype=float).reshape(len(self.lower))
        if not np.isfinite(q).all():
            raise ValueError("Non-finite joints")
        transform = np.eye(4)
        origins, axes, points = [], [], [np.zeros(3)]
        index = 0
        for origin, axis, kind in self.chain:
            transform = transform @ origin
            points.append(transform[:3, 3].copy())
            if kind == "fixed":
                continue
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ axis)
            x, y, z = axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            rotation = np.eye(4)
            rotation[:3, :3] += np.sin(q[index]) * skew + (1 - np.cos(q[index])) * skew @ skew
            transform = transform @ rotation
            index += 1
        transform = transform @ self.tool
        points.append(transform[:3, 3].copy())
        return transform, origins, axes, np.asarray(points)

    def forward(self, joints):
        return self._evaluate(joints)[0]

    def link_positions(self, joints):
        return self._evaluate(joints)[3]

    def jacobian(self, joints):
        pose, origins, axes, _ = self._evaluate(joints)
        return np.vstack(
            (
                np.array(
                    [np.cross(a, pose[:3, 3] - p) for a, p in zip(axes, origins, strict=True)]
                ).T,
                np.array(axes).T,
            )
        )
