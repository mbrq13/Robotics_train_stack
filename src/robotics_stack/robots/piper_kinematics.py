"""Piper arm forward kinematics and geometric Jacobian for teleoperation."""

from __future__ import annotations

import numpy as np

PIPER_LOWER_RAD = np.array((-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944))
PIPER_UPPER_RAD = np.array((2.618, 3.14, 0.0, 1.745, 1.22, 2.0944))

_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),
)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _origin_transform(
    translation: tuple[float, float, float], rpy: tuple[float, float, float]
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_z(rpy[2]) @ _rotation_y(rpy[1]) @ _rotation_x(rpy[0])
    transform[:3, 3] = translation
    return transform


class PiperArmKinematics:
    """Six revolute joints ending at the Piper tool center point."""

    def __init__(self, *, lateral_offset_m: float = 0.0) -> None:
        self._base = np.eye(4, dtype=np.float64)
        self._base[1, 3] = lateral_offset_m
        self._origins = tuple(_origin_transform(position, rpy) for position, rpy in _JOINT_ORIGINS)
        self._tool = _origin_transform((0.0, 0.0, 0.145), (0.0, 0.0, 0.0))

    def _chain(self, joints: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        q = np.asarray(joints, dtype=np.float64).reshape(6)
        if not np.isfinite(q).all():
            raise ValueError("Piper joints must be finite")
        transform = self._base.copy()
        origins: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        for origin, angle in zip(self._origins, q, strict=True):
            transform = transform @ origin
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, 2].copy())
            rotation = np.eye(4, dtype=np.float64)
            rotation[:3, :3] = _rotation_z(float(angle))
            transform = transform @ rotation
        return transform @ self._tool, origins, axes

    def forward(self, joints: np.ndarray) -> np.ndarray:
        """Return the tool transform for six joint angles."""
        transform, _, _ = self._chain(joints)
        return transform

    def jacobian(self, joints: np.ndarray) -> np.ndarray:
        """Return the geometric position-and-orientation Jacobian."""
        tool, origins, axes = self._chain(joints)
        endpoint = tool[:3, 3]
        result = np.empty((6, 6), dtype=np.float64)
        for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
            result[:3, index] = np.cross(axis, endpoint - origin)
            result[3:, index] = axis
        return result
