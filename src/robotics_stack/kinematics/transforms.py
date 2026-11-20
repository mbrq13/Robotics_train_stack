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
    quaternion = quaternion_xyzw_from_rotation(rotation)
    norm = np.linalg.norm(quaternion[:3])
    if norm < 1e-10:
        return 2.0 * quaternion[:3]
    angle = 2.0 * np.arctan2(norm, quaternion[3])
    return quaternion[:3] * (angle / norm)


def anchored_target(
    source_anchor: np.ndarray,
    source_current: np.ndarray,
    robot_anchor: np.ndarray,
    *,
    source_to_robot_rotation: np.ndarray,
    translation_scale: float,
    max_displacement_m: float,
) -> np.ndarray:
    """Map anchored motion between world frames, independent of initial tool axes.

    ``source_to_robot_rotation`` maps source-world vectors to robot-base vectors.
    The source rotation delta acts on the left of the anchored robot rotation.
    """
    if not all(np.isfinite(v) and v > 0 for v in (translation_scale, max_displacement_m)):
        raise ValueError("translation scale and reach bound must be positive")
    world_map = np.asarray(source_to_robot_rotation, dtype=np.float64).reshape(3, 3)
    if (
        not np.isfinite(world_map).all()
        or not np.allclose(world_map.T @ world_map, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(world_map), 1.0, atol=1e-6)
    ):
        raise ValueError("source_to_robot_rotation must be a proper rotation")
    anchor = pose_matrix(source_anchor)
    current = pose_matrix(source_current)
    robot = pose_matrix(robot_anchor)
    displacement = world_map @ (current[:3, 3] - anchor[:3, 3]) * translation_scale
    distance = float(np.linalg.norm(displacement))
    if distance > max_displacement_m:
        displacement *= max_displacement_m / distance
    target = robot.copy()
    target[:3, 3] += displacement
    source_rotation_delta = current[:3, :3] @ anchor[:3, :3].T
    target[:3, :3] = world_map @ source_rotation_delta @ world_map.T @ robot[:3, :3]
    return matrix_pose(target)


def origin_transform(translation, rpy):
    """URDF origin: translation and fixed-axis roll, pitch, yaw."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    result[:3, 3] = translation
    return result
