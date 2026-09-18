"""Small SE(3) utilities used by device-independent VR retargeting."""

from __future__ import annotations

import numpy as np


def rotation_from_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    """Return a proper rotation matrix from an ``[x, y, z, w]`` quaternion."""
    x, y, z, w = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm < 1e-12:
        raise ValueError("a pose quaternion must have non-zero norm")
    x, y, z, w = np.asarray((x, y, z, w), dtype=np.float64) / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def quaternion_xyzw_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion with a non-negative scalar component."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        other = (axis + 1) % 3
        last = (axis + 2) % 3
        scale = 2.0 * np.sqrt(1.0 + matrix[axis, axis] - matrix[other, other] - matrix[last, last])
        quaternion = np.zeros(4, dtype=np.float64)
        quaternion[axis] = 0.25 * scale
        quaternion[other] = (matrix[axis, other] + matrix[other, axis]) / scale
        quaternion[last] = (matrix[axis, last] + matrix[last, axis]) / scale
        quaternion[3] = (matrix[last, other] - matrix[other, last]) / scale
    quaternion /= np.linalg.norm(quaternion)
    return -quaternion if quaternion[3] < 0.0 else quaternion


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a position plus xyzw quaternion to a homogeneous transform."""
    pose7 = np.asarray(pose, dtype=np.float64).reshape(7)
    if not np.isfinite(pose7).all():
        raise ValueError("a pose must contain finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_quaternion_xyzw(pose7[3:])
    transform[:3, 3] = pose7[:3]
    return transform


def matrix_pose(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to position plus xyzw quaternion."""
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.concatenate((matrix[:3, 3], quaternion_xyzw_from_rotation(matrix[:3, :3])))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the shortest axis-angle vector for a relative rotation."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        )
    )
    if angle < 1e-7:
        return 0.5 * skew
    return skew * (angle / (2.0 * np.sin(angle)))


def anchored_target(
    source_anchor: np.ndarray,
    source_current: np.ndarray,
    robot_anchor: np.ndarray,
    *,
    source_to_robot_rotation: np.ndarray,
    translation_scale: float,
    max_displacement_m: float,
) -> np.ndarray:
    """Apply the tracked controller motion relative to an explicit anchor."""
    if not translation_scale > 0.0 or not max_displacement_m > 0.0:
        raise ValueError("translation scale and reach bound must be positive")
    adapter = np.eye(4, dtype=np.float64)
    adapter[:3, :3] = np.asarray(source_to_robot_rotation, dtype=np.float64).reshape(3, 3)
    delta = np.linalg.inv(pose_matrix(source_anchor)) @ pose_matrix(source_current)
    mapped_delta = adapter @ delta @ np.linalg.inv(adapter)
    mapped_delta[:3, 3] *= translation_scale
    target = pose_matrix(robot_anchor) @ mapped_delta
    displacement = target[:3, 3] - np.asarray(robot_anchor, dtype=np.float64).reshape(7)[:3]
    distance = float(np.linalg.norm(displacement))
    if distance > max_displacement_m:
        target[:3, 3] -= displacement * (1.0 - max_displacement_m / distance)
    return matrix_pose(target)
