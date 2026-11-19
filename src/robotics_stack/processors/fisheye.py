"""Approximate equidistant-fisheye rectification for Dream camera previews."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, degrees, radians, tan
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class FisheyeRectification:
    """Geometry for mapping a declared fisheye FOV into a virtual pinhole camera."""

    source_hfov_deg: float = 200.0
    source_vfov_deg: float = 150.0
    target_hfov_deg: float = 84.0
    output_width: int = 640
    output_height: int = 360

    def validate(self) -> None:
        if not 0 < self.source_hfov_deg <= 220:
            raise ValueError("source_hfov_deg must be in (0, 220]")
        if not 0 < self.source_vfov_deg <= 220:
            raise ValueError("source_vfov_deg must be in (0, 220]")
        if not 0 < self.target_hfov_deg < 170:
            raise ValueError("target_hfov_deg must be in (0, 170)")
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("output dimensions must be positive")

    @property
    def target_vfov_deg(self) -> float:
        focal = self.output_width / (2 * tan(radians(self.target_hfov_deg) / 2))
        return degrees(2 * atan(self.output_height / (2 * focal)))


@dataclass(frozen=True)
class CameraPreprocessProfile:
    """Camera-owned preprocessing selection, independent of policy configuration."""

    kind: str
    roles: tuple[str, ...]
    rectification: FisheyeRectification


def load_camera_preprocess_profile(path: str | Path) -> CameraPreprocessProfile:
    """Load one explicit camera preprocessing profile from YAML."""
    profile_path = Path(path).expanduser()
    with profile_path.open(encoding="utf-8") as stream:
        value: Any = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{profile_path} must contain a mapping")
    kind = str(value.get("type", ""))
    if kind != "equidistant_fisheye":
        raise ValueError(f"unsupported camera preprocessing type {kind!r}")
    roles_value = value.get("roles", ["left", "right"])
    if not isinstance(roles_value, list) or not roles_value:
        raise ValueError("camera preprocessing roles must be a non-empty list")
    roles = tuple(str(role) for role in roles_value)
    if len(set(roles)) != len(roles):
        raise ValueError("camera preprocessing roles must be unique")
    rectification = FisheyeRectification(
        source_hfov_deg=float(value.get("source_hfov_deg", 200.0)),
        source_vfov_deg=float(value.get("source_vfov_deg", 150.0)),
        target_hfov_deg=float(value.get("target_hfov_deg", 84.0)),
        output_width=int(value.get("output_width", 640)),
        output_height=int(value.get("output_height", 360)),
    )
    rectification.validate()
    return CameraPreprocessProfile(kind=kind, roles=roles, rectification=rectification)


def center_crop_16_9(frame: np.ndarray, *, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    """Center-crop a frame to 16:9, then resize it to ``size``."""
    height, width = frame.shape[:2]
    target_aspect = 16 / 9
    if width / height > target_aspect:
        crop_width = round(height * target_aspect)
        x0 = (width - crop_width) // 2
        cropped = frame[:, x0 : x0 + crop_width]
    else:
        crop_height = round(width / target_aspect)
        y0 = (height - crop_height) // 2
        cropped = frame[y0 : y0 + crop_height]
    return cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)


def approximate_rectification_maps(
    source_size: tuple[int, int], config: FisheyeRectification
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build output-to-input maps for a centered equidistant fisheye lens.

    The advertised horizontal and vertical FOVs independently determine the
    source focal lengths. This is intentionally an approximation for previews;
    calibrated OpenCV fisheye intrinsics should replace it for deployment.
    """
    config.validate()
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")

    source_fx = source_width / radians(config.source_hfov_deg)
    source_fy = source_height / radians(config.source_vfov_deg)
    source_cx = (source_width - 1) / 2
    source_cy = (source_height - 1) / 2
    target_focal = config.output_width / (2 * tan(radians(config.target_hfov_deg) / 2))
    target_cx = (config.output_width - 1) / 2
    target_cy = (config.output_height - 1) / 2

    output_x, output_y = np.meshgrid(
        np.arange(config.output_width, dtype=np.float32),
        np.arange(config.output_height, dtype=np.float32),
    )
    ray_x = (output_x - target_cx) / target_focal
    ray_y = (output_y - target_cy) / target_focal
    radius = np.hypot(ray_x, ray_y)
    theta = np.arctan(radius)
    scale = np.divide(theta, radius, out=np.ones_like(theta), where=radius > 1e-8)
    map_x = (source_cx + source_fx * scale * ray_x).astype(np.float32)
    map_y = (source_cy + source_fy * scale * ray_y).astype(np.float32)

    valid = (
        (map_x >= 0)
        & (map_x <= source_width - 1)
        & (map_y >= 0)
        & (map_y <= source_height - 1)
    )
    metrics = {
        "source_fx_px": float(source_fx),
        "source_fy_px": float(source_fy),
        "target_focal_px": float(target_focal),
        "target_vfov_deg": config.target_vfov_deg,
        "mapped_source_width_px": float(map_x.max() - map_x.min()),
        "mapped_source_height_px": float(map_y.max() - map_y.min()),
        "valid_output_fraction": float(np.mean(valid)),
    }
    return map_x, map_y, metrics


def rectify_equidistant_fisheye(
    frame: np.ndarray, config: FisheyeRectification
) -> tuple[np.ndarray, dict[str, float]]:
    """Rectify ``frame`` into the configured virtual pinhole view."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape [height, width, 3]")
    map_x, map_y, metrics = approximate_rectification_maps(
        (frame.shape[1], frame.shape[0]), config
    )
    rectified = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rectified, metrics
