#!/usr/bin/env python3
"""Preview an approximate SVPRO-fisheye conversion into Dream's visual geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from robotics_stack.processors.fisheye import (
    FisheyeRectification,
    center_crop_16_9,
    rectify_equidistant_fisheye,
)


def _video_frame(path: Path, timestamp_s: float) -> np.ndarray:
    """Decode the frame nearest ``timestamp_s`` with PyAV's software decoder."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        best: tuple[float, np.ndarray] | None = None
        for frame in container.decode(stream):
            frame_time = float(frame.time or 0.0)
            distance = abs(frame_time - timestamp_s)
            image = frame.to_ndarray(format="bgr24")
            if best is None or distance < best[0]:
                best = (distance, image)
            if frame_time >= timestamp_s:
                break
    if best is None:
        raise RuntimeError(f"video has no decodable frames: {path}")
    return best[1]


def _read(path: Path, timestamp_s: float) -> np.ndarray:
    if path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
        return _video_frame(path, timestamp_s)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return frame


def _fit_panel(image: np.ndarray, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 18, np.uint8)
    x0 = (width - resized.shape[1]) // 2
    y0 = (height - resized.shape[0]) // 2
    panel[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return panel


def _label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(result, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(result, subtitle, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image or video")
    parser.add_argument("--timestamp", type=float, default=8.0, help="Video time in seconds")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--source-hfov", type=float, default=200.0)
    parser.add_argument("--source-vfov", type=float, default=150.0)
    parser.add_argument("--target-hfov", type=float, default=84.0)
    args = parser.parse_args()

    frame = _read(args.input, args.timestamp)
    config = FisheyeRectification(
        source_hfov_deg=args.source_hfov,
        source_vfov_deg=args.source_vfov,
        target_hfov_deg=args.target_hfov,
    )
    crop = center_crop_16_9(frame)
    rectified, metrics = rectify_equidistant_fisheye(frame, config)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "raw.png"), frame)
    cv2.imwrite(str(args.output / "center_crop_16_9.png"), crop)
    cv2.imwrite(str(args.output / "rectified_84deg.png"), rectified)

    raw_panel = _label(
        _fit_panel(frame),
        "Original dataset frame",
        (
            f"{frame.shape[1]}x{frame.shape[0]} | advertised "
            f"{args.source_hfov:.0f}x{args.source_vfov:.0f} deg"
        ),
    )
    crop_panel = _label(
        crop,
        "Current center crop",
        "640x360 | aspect fixed, fisheye distortion remains",
    )
    rectified_panel = _label(
        rectified,
        "Approximate Dream-like view",
        f"640x360 | pinhole {args.target_hfov:.0f}x{config.target_vfov_deg:.1f} deg",
    )
    comparison = np.hstack((raw_panel, crop_panel, rectified_panel))
    cv2.imwrite(str(args.output / "comparison.png"), comparison)

    report = {
        "input": str(args.input),
        "timestamp_s": args.timestamp,
        "assumption": "centered equidistant fisheye; no measured K/D calibration",
        "source_size": [frame.shape[1], frame.shape[0]],
        "source_hfov_deg": args.source_hfov,
        "source_vfov_deg": args.source_vfov,
        "target_size": [config.output_width, config.output_height],
        "target_hfov_deg": args.target_hfov,
        **metrics,
        "deployment_ready": False,
        "next_step": "replace advertised FOV with per-camera OpenCV fisheye calibration",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
