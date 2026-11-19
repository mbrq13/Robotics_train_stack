"""Open-loop Dream evaluation on a recorded LeRobot v3 episode."""

from __future__ import annotations

import csv
import html
import json
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robotics_stack.contracts import Observation
from robotics_stack.policies.dream1 import Dream1Config, Dream1Policy
from robotics_stack.processors.fisheye import (
    CameraPreprocessProfile,
    approximate_rectification_maps,
    load_camera_preprocess_profile,
)
from robotics_stack.robots.piper_kinematics import PiperArmKinematics

CAMERAS = ("left", "top", "right")
JOINT_INDICES = tuple(range(6)) + tuple(range(7, 13))
GRIPPER_INDICES = (6, 13)
JOINT_NAMES = tuple(f"L{index}" for index in range(1, 7)) + tuple(
    f"R{index}" for index in range(1, 7)
)
JOINT_LOWER = np.asarray([-2.618, 0, -2.967, -1.745, -1.22, -2.0944] * 2)
JOINT_UPPER = np.asarray([2.618, 3.14, 0, 1.745, 1.22, 2.0944] * 2)


def _download(dataset_id: str, relative: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/datasets/"
        + urllib.parse.quote(dataset_id, safe="/")
        + "/resolve/main/"
        + urllib.parse.quote(relative, safe="/")
    )
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {relative} ...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    partial.replace(destination)
    return destination


def _episode_metadata(dataset_id: str, cache: Path, episode: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = _download(
        dataset_id,
        "meta/episodes/chunk-000/file-000.parquet",
        cache / dataset_id / "meta" / "episodes.parquet",
    )
    rows = pq.read_table(path).to_pylist()
    found = next((row for row in rows if int(row["episode_index"]) == episode), None)
    if found is None:
        raise ValueError(f"dataset has no episode {episode}")
    return found


def _episode_data(dataset_id: str, cache: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    chunk = int(metadata["data/chunk_index"])
    index = int(metadata["data/file_index"])
    relative = f"data/chunk-{chunk:03d}/file-{index:03d}.parquet"
    path = _download(dataset_id, relative, cache / dataset_id / relative)
    return pq.read_table(path).to_pylist()


def _video_paths(
    dataset_id: str, cache: Path, metadata: dict[str, Any]
) -> dict[str, tuple[Path, float]]:
    result = {}
    for camera in CAMERAS:
        prefix = f"videos/observation.images.{camera}"
        chunk = int(metadata[f"{prefix}/chunk_index"])
        index = int(metadata[f"{prefix}/file_index"])
        relative = f"{prefix}/chunk-{chunk:03d}/file-{index:03d}.mp4"
        path = _download(dataset_id, relative, cache / dataset_id / relative)
        result[camera] = (path, float(metadata[f"{prefix}/from_timestamp"]))
    return result


class _VideoReader:
    def __init__(self, path: Path) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("install the dream-replay extra to decode dataset video") from exc
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]

    def at(self, timestamp_s: float) -> np.ndarray:
        timestamp_s = max(0.0, timestamp_s)
        offset = round(timestamp_s / float(self.stream.time_base))
        self.container.seek(offset, stream=self.stream, backward=True)
        closest = None
        for frame in self.container.decode(self.stream):
            closest = frame
            if frame.time is not None and frame.time >= timestamp_s - 1e-4:
                return frame.to_ndarray(format="bgr24")
        if closest is None:
            raise RuntimeError(f"cannot decode video frame at {timestamp_s:.3f}s")
        return closest.to_ndarray(format="bgr24")

    def close(self) -> None:
        self.container.close()


def _policy_frame(frame: np.ndarray, camera: str, mode: str) -> np.ndarray:
    if mode in {"deployment_crop", "deployment_crop_repeat"}:
        height, width = frame.shape[:2]
        target_ratio = 16 / 9
        crop_height = min(height, round(width / target_ratio))
        start = (height - crop_height) // 2
        cropped = frame[start : start + crop_height]
        return cv2.resize(cropped, (640, 360), interpolation=cv2.INTER_AREA)
    if mode != "full_fov_letterbox":
        raise ValueError(f"unknown FOV mode {mode}")
    height, width = frame.shape[:2]
    scale = min(640 / width, 360 / height)
    resized = cv2.resize(
        frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    y = (360 - resized.shape[0]) // 2
    x = (640 - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


class _ReplayFramePreprocessor:
    """Apply an optional camera-owned profile to recorded frames."""

    def __init__(self, profile_path: str | Path | None) -> None:
        self.path = str(profile_path) if profile_path is not None else None
        self.profile: CameraPreprocessProfile | None = (
            load_camera_preprocess_profile(profile_path) if profile_path is not None else None
        )
        self._maps: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def apply(self, frame: np.ndarray, camera: str) -> np.ndarray:
        if self.profile is None or camera not in self.profile.roles:
            return _policy_frame(frame, camera, "deployment_crop")
        key = (camera, frame.shape[1], frame.shape[0])
        maps = self._maps.get(key)
        if maps is None:
            map_x, map_y, _metrics = approximate_rectification_maps(
                (frame.shape[1], frame.shape[0]), self.profile.rectification
            )
            maps = (map_x, map_y)
            self._maps[key] = maps
        return cv2.remap(
            frame,
            maps[0],
            maps[1],
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
        )


def _jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cannot encode replay frame")
    return encoded.tobytes()


def _query_indices(length: int, count: int, history_s: float, fps: float) -> list[int]:
    if count < 1:
        raise ValueError("queries must be positive")
    first = min(length - 1, max(round(history_s * fps), round(length * 0.15)))
    last = max(first, round((length - 1) * 0.85))
    return sorted(set(np.linspace(first, last, count).round().astype(int).tolist()))


def _reference_chunk(
    rows: list[dict[str, Any]], index: int, length: int, action_hz: float
) -> np.ndarray:
    times = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
    desired = times[index] + np.arange(length, dtype=np.float64) / action_hz
    desired = np.clip(desired, times[0], times[-1])
    return np.stack([np.interp(desired, times, actions[:, column]) for column in range(14)], axis=1)


def _metrics(predicted: np.ndarray, reference: np.ndarray, anchor: np.ndarray) -> dict[str, Any]:
    rows = min(len(predicted), len(reference))
    predicted, reference = predicted[:rows], reference[:rows]
    joint_error = predicted[:, JOINT_INDICES] - reference[:, JOINT_INDICES]
    initial = predicted[0, JOINT_INDICES] - anchor[list(JOINT_INDICES)]
    per_joint = np.rad2deg(np.mean(np.abs(joint_error), axis=0))
    steps = np.diff(
        np.concatenate((anchor[list(JOINT_INDICES)][None], predicted[:, JOINT_INDICES])), axis=0
    )
    joints = predicted[:, JOINT_INDICES]
    violations = np.count_nonzero((joints < JOINT_LOWER) | (joints > JOINT_UPPER))
    return {
        "joint_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(joint_error**2)))),
        "joint_mae_deg": float(np.rad2deg(np.mean(np.abs(joint_error)))),
        "first_target_delta_deg_max": float(np.rad2deg(np.max(np.abs(initial)))),
        "gripper_mae": float(
            np.mean(np.abs(predicted[:, GRIPPER_INDICES] - reference[:, GRIPPER_INDICES]))
        ),
        "predicted_joint_step_deg_max": float(
            np.rad2deg(np.max(np.abs(np.diff(predicted[:, JOINT_INDICES], axis=0)), initial=0))
        ),
        "predicted_joint_speed_rad_s_max": float(np.max(np.abs(steps)) * 20.0),
        "joint_limit_violation_count": int(violations),
        "joint_mae_deg_by_name": dict(zip(JOINT_NAMES, per_joint.tolist(), strict=True)),
    }


def _contact_sheet(samples: list[tuple[str, int, dict[str, np.ndarray]]], path: Path) -> None:
    rows = []
    for mode, index, images in samples:
        panels = []
        for camera in CAMERAS:
            panel = images[camera].copy()
            cv2.putText(
                panel, f"{mode} | frame {index} | {camera}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
            )
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def _write_html(report: dict[str, Any], path: Path) -> None:
    query_rows = "".join(
        "<tr>"
        + f"<td>{item['mode']}</td><td>{item['frame_index']}</td>"
        + f"<td>{item['metrics']['joint_rmse_deg']:.2f}°</td>"
        + f"<td>{item['metrics']['joint_mae_deg']:.2f}°</td>"
        + f"<td>{item['metrics']['first_target_delta_deg_max']:.2f}°</td>"
        + f"<td>{item['metrics']['gripper_mae']:.3f}</td></tr>"
        for item in report["queries"]
    )
    fov = report.get("fov", {})
    body = f"""<!doctype html><meta charset='utf-8'><title>Dream replay report</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:7px}}
img{{max-width:100%}}</style>
<h1>Dream-1 open-loop dataset replay</h1>
<p>Dataset: {html.escape(report['dataset'])}; episode {report['episode']}; task:
{html.escape(report['task'])}.</p>
<p>This compares local Dream action chunks with the recorded continuation. It does not simulate
the scene response to Dream's actions and therefore cannot measure task success.</p>
<h2>Camera inputs</h2><img src='fov_contact_sheet.jpg'>
<h2>Angular comparison</h2><table><tr><th>FOV</th><th>Frame</th><th>Joint RMSE</th>
<th>Joint MAE</th><th>Largest first target delta</th><th>Gripper MAE</th></tr>{query_rows}</table>
<p><img src='angle_error_heatmap.jpg'></p>
<h2>FOV sensitivity</h2><pre>{html.escape(json.dumps(fov, indent=2))}</pre>
<p>Raw chunks and reference actions are in <code>actions.csv</code>; machine-readable results are
in <code>report.json</code>.</p>"""
    path.write_text(body, encoding="utf-8")


def _angle_heatmap(report: dict[str, Any], path: Path) -> None:
    queries = report["queries"]
    cell_w, cell_h, label_w = 76, 42, 270
    canvas = np.full(
        ((len(queries) + 1) * cell_h, label_w + len(JOINT_NAMES) * cell_w, 3), 255, np.uint8
    )
    for column, name in enumerate(JOINT_NAMES):
        cv2.putText(
            canvas, name, (label_w + column * cell_w + 22, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
    for row_index, query in enumerate(queries, start=1):
        label = f"{query['mode']} f{query['frame_index']}"
        cv2.putText(
            canvas, label, (5, row_index * cell_h + 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA,
        )
        values = query["metrics"]["joint_mae_deg_by_name"]
        for column, name in enumerate(JOINT_NAMES):
            value = float(values[name])
            intensity = min(1.0, value / 30.0)
            color = (round(255 * (1 - intensity)), round(255 * (1 - intensity)), 255)
            x0, y0 = label_w + column * cell_w, row_index * cell_h
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), color, -1)
            cv2.putText(
                canvas, f"{value:.1f}", (x0 + 14, y0 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)


def run_dream_dataset_replay(
    *, config_path: str | Path, dataset_id: str, episode: int, queries: int, task: str,
    output: Path, cache: Path, fov_ablation: bool,
) -> dict[str, Any]:
    config = replace(
        Dream1Config.load(config_path),
        task=task.strip(),
        require_tcp_confirmation=False,
        dry_run=True,
    )
    config.validate()
    metadata = _episode_metadata(dataset_id, cache, episode)
    rows = _episode_data(dataset_id, cache, metadata)
    paths = _video_paths(dataset_id, cache, metadata)
    readers = {camera: _VideoReader(path) for camera, (path, _start) in paths.items()}
    output.mkdir(parents=True, exist_ok=True)
    indices = _query_indices(len(rows), queries, (config.obs_window - 1) / config.obs_fps, 30)
    modes = ["deployment_crop"]
    if fov_ablation:
        # A repeated identical input separates sampling/session variation from FOV sensitivity.
        modes += ["deployment_crop_repeat", "full_fov_letterbox"]
    results: list[dict[str, Any]] = []
    chunks: dict[tuple[str, int], np.ndarray] = {}
    contact: list[tuple[str, int, dict[str, np.ndarray]]] = []
    policy = Dream1Policy(config)
    base_clock = time.monotonic()
    try:
        for mode in modes:
            for index in indices:
                print(f"Dream query: {mode}, frame {index}/{len(rows) - 1}", flush=True)
                policy.reset_episode()
                query_t = float(rows[index]["timestamp"])
                last_images = {}
                for history_index in range(config.obs_window):
                    timestamp = query_t - (config.obs_window - 1 - history_index) / config.obs_fps
                    timestamp = max(0.0, timestamp)
                    distances = [abs(float(row["timestamp"]) - timestamp) for row in rows]
                    source_index = int(np.argmin(distances))
                    images = {}
                    for camera in CAMERAS:
                        path, video_start = paths[camera]
                        del path
                        frame_timestamp = video_start + float(rows[source_index]["timestamp"])
                        raw = readers[camera].at(frame_timestamp)
                        transformed = _policy_frame(raw, camera, mode)
                        images[camera] = _jpeg(transformed)
                        if history_index == config.obs_window - 1:
                            last_images[camera] = transformed
                    policy.set_observation(
                        Observation(
                            observation_id=source_index,
                            station_monotonic_ns=int((base_clock + timestamp) * 1e9),
                            state=tuple(
                                float(value)
                                for value in rows[source_index]["observation.state"]
                            ),
                            images=images,
                        )
                    )
                anchor = np.asarray(rows[index]["observation.state"], dtype=np.float64)
                started = time.perf_counter()
                predicted = policy.get_chunk(tuple(anchor.tolist()))
                elapsed = time.perf_counter() - started
                reference = _reference_chunk(rows, index, len(predicted), config.action_hz)
                chunks[(mode, index)] = predicted
                result = {
                    "mode": mode, "frame_index": index, "timestamp_s": query_t,
                    "inference_s": elapsed, "chunk_rows": len(predicted),
                    "within_live_timeout": elapsed <= config.action_timeout_s,
                    "metrics": _metrics(predicted, reference, anchor),
                }
                results.append(result)
                if mode != "deployment_crop_repeat":
                    contact.append((mode, index, last_images))
    finally:
        policy.close()
        for reader in readers.values():
            reader.close()

    fov: dict[str, Any] = {}
    if fov_ablation:
        differences = []
        repeat_differences = []
        for index in indices:
            crop = chunks[("deployment_crop", index)][:, JOINT_INDICES]
            full = chunks[("full_fov_letterbox", index)][:, JOINT_INDICES]
            repeated = chunks[("deployment_crop_repeat", index)][:, JOINT_INDICES]
            differences.append(np.abs(crop - full))
            repeat_differences.append(np.abs(crop - repeated))
        delta = np.concatenate(differences)
        repeat_delta = np.concatenate(repeat_differences)
        fov_mae = float(np.rad2deg(np.mean(delta)))
        repeat_mae = float(np.rad2deg(np.mean(repeat_delta)))
        fov = {
            "crop_vs_full_joint_mae_deg": fov_mae,
            "joint_action_p95_deg": float(np.rad2deg(np.percentile(delta, 95))),
            "joint_action_max_deg": float(np.rad2deg(np.max(delta))),
            "same_crop_repeat_joint_mae_deg": repeat_mae,
            "fov_excess_over_repeat_mae_deg": max(0.0, fov_mae - repeat_mae),
            "interpretation": (
                "Large deltas mean the policy is sensitive to crop versus full-frame FOV."
            ),
        }
    report = {
        "dataset": dataset_id, "episode": episode, "task": task,
        "open_loop": True, "query_frames": indices, "queries": results, "fov": fov,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "actions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "query_frame", "chunk_row", *[f"action_{i}" for i in range(14)]])
        for (mode, index), chunk in chunks.items():
            for chunk_row, action in enumerate(chunk):
                writer.writerow([mode, index, chunk_row, *action.tolist()])
            reference = _reference_chunk(rows, index, len(chunk), config.action_hz)
            for chunk_row, action in enumerate(reference):
                writer.writerow([f"reference_for_{mode}", index, chunk_row, *action.tolist()])
    _contact_sheet(contact, output / "fov_contact_sheet.jpg")
    _angle_heatmap(report, output / "angle_error_heatmap.jpg")
    _write_html(report, output / "report.html")
    return report


def _interpolate_rows(times: np.ndarray, values: np.ndarray, desired: np.ndarray) -> np.ndarray:
    desired = np.clip(desired, times[0], times[-1])
    return np.stack(
        [np.interp(desired, times, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def _draw_arm(
    canvas: np.ndarray,
    points: np.ndarray,
    *,
    panel: tuple[int, int, int, int],
    axes: tuple[int, int],
    ranges: tuple[tuple[float, float], tuple[float, float]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x0, y0, width, height = panel
    horizontal, vertical = axes
    (h_lo, h_hi), (v_lo, v_hi) = ranges
    pixels = []
    for point in points:
        x = x0 + round((point[horizontal] - h_lo) / (h_hi - h_lo) * width)
        y = y0 + height - round((point[vertical] - v_lo) / (v_hi - v_lo) * height)
        pixels.append((x, y))
    cv2.polylines(canvas, [np.asarray(pixels, np.int32)], False, color, thickness, cv2.LINE_AA)
    for pixel in pixels:
        cv2.circle(canvas, pixel, thickness + 2, color, -1, cv2.LINE_AA)


def _render_shadow_video(
    *,
    readers: dict[str, _VideoReader],
    paths: dict[str, tuple[Path, float]],
    sim_times: np.ndarray,
    simulated: np.ndarray,
    reference: np.ndarray,
    output: Path,
    frame_preprocessor: _ReplayFramePreprocessor,
    fps: float = 10.0,
) -> None:
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create shadow video {output}")
    left_fk = PiperArmKinematics(lateral_offset_m=0.30)
    right_fk = PiperArmKinematics(lateral_offset_m=-0.30)
    frame_times = np.arange(sim_times[0], sim_times[-1] + 1e-6, 1.0 / fps)
    sim_video = _interpolate_rows(sim_times, simulated, frame_times)
    ref_video = _interpolate_rows(sim_times, reference, frame_times)
    try:
        for frame_number, (timestamp, state, expected) in enumerate(
            zip(frame_times, sim_video, ref_video, strict=True)
        ):
            canvas = np.full((720, 1280, 3), 20, np.uint8)
            for camera_index, camera in enumerate(CAMERAS):
                _path, video_start = paths[camera]
                raw = readers[camera].at(video_start + float(timestamp))
                image = frame_preprocessor.apply(raw, camera)
                image = cv2.resize(image, (426, 240), interpolation=cv2.INTER_AREA)
                x = camera_index * 427
                canvas[:240, x : x + 426] = image
                cv2.putText(
                    canvas, camera, (x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA,
                )
            top_panel = (35, 285, 570, 385)
            front_panel = (675, 285, 570, 385)
            for panel in (top_panel, front_panel):
                cv2.rectangle(
                    canvas, (panel[0], panel[1]),
                    (panel[0] + panel[2], panel[1] + panel[3]), (80, 80, 80), 1,
                )
            pairs = (
                (left_fk, state[:6], expected[:6], (0, 220, 255)),
                (right_fk, state[7:13], expected[7:13], (255, 180, 0)),
            )
            for kinematics, dream_joints, reference_joints, color in pairs:
                dream_points = kinematics.link_positions(dream_joints)
                reference_points = kinematics.link_positions(reference_joints)
                for panel, axes, ranges in (
                    (top_panel, (0, 1), ((-0.10, 0.75), (-0.72, 0.72))),
                    (front_panel, (1, 2), ((-0.72, 0.72), (-0.05, 0.75))),
                ):
                    _draw_arm(
                        canvas, reference_points, panel=panel, axes=axes, ranges=ranges,
                        color=(105, 105, 105), thickness=2,
                    )
                    _draw_arm(
                        canvas, dream_points, panel=panel, axes=axes, ranges=ranges,
                        color=color, thickness=4,
                    )
            error = np.rad2deg(
                np.sqrt(np.mean((state[list(JOINT_INDICES)] - expected[list(JOINT_INDICES)]) ** 2))
            )
            cv2.putText(
                canvas, "Top view: X/Y", (45, 315), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (240, 240, 240), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "Front view: Y/Z", (685, 315), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (240, 240, 240), 2, cv2.LINE_AA,
            )
            label = (
                f"SHADOW ROLLOUT | t={timestamp:.2f}s | joint RMSE={error:.1f} deg | "
                "gray=recorded, color=Dream"
            )
            cv2.putText(
                canvas, label, (22, 704), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(canvas)
            if frame_number % round(fps * 5) == 0:
                print(f"Rendering shadow video: {timestamp:.1f}/{sim_times[-1]:.1f}s", flush=True)
    finally:
        writer.release()


def run_dream_shadow_rollout(
    *,
    config_path: str | Path,
    dataset_id: str,
    episode: int,
    task: str,
    output: Path,
    cache: Path,
    camera_preprocess_profile: str | Path | None = None,
) -> dict[str, Any]:
    """Advance simulated Piper joints with Dream while replaying recorded camera observations."""
    config = replace(
        Dream1Config.load(config_path),
        task=task.strip(),
        require_tcp_confirmation=False,
        dry_run=True,
    )
    config.validate()
    metadata = _episode_metadata(dataset_id, cache, episode)
    rows = _episode_data(dataset_id, cache, metadata)
    paths = _video_paths(dataset_id, cache, metadata)
    readers = {camera: _VideoReader(path) for camera, (path, _start) in paths.items()}
    frame_preprocessor = _ReplayFramePreprocessor(camera_preprocess_profile)
    output.mkdir(parents=True, exist_ok=True)
    dataset_times = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    dataset_states = np.asarray([row["observation.state"] for row in rows], dtype=np.float64)
    start_s = 0.0
    end_s = float(dataset_times[-1])
    state = _interpolate_rows(dataset_times, dataset_states, np.asarray([start_s]))[0]
    sim_times = [start_s]
    simulated = [state.copy()]
    latencies = []
    query_times = []
    last_observation_time = -1.0
    next_sample_time = 0.0
    observation_id = 0
    policy = Dream1Policy(config)
    base_clock = time.monotonic()
    try:
        policy.reset_episode()
        while sim_times[-1] < end_s - 1e-6:
            query_time = sim_times[-1]
            query_times.append(query_time)
            print(f"Dream shadow query at {query_time:.2f}/{end_s:.2f}s", flush=True)
            history_times = []
            while next_sample_time <= query_time + 1e-6:
                if next_sample_time > last_observation_time + 1e-6:
                    history_times.append(next_sample_time)
                next_sample_time += 1 / config.obs_fps
            if query_time > last_observation_time + 1e-6 and (
                not history_times or abs(history_times[-1] - query_time) > 1e-6
            ):
                history_times.append(query_time)
            for history_time in history_times:
                history_state = _interpolate_rows(
                    np.asarray(sim_times), np.asarray(simulated), np.asarray([history_time])
                )[0]
                images = {}
                for camera in CAMERAS:
                    _path, video_start = paths[camera]
                    raw = readers[camera].at(video_start + history_time)
                    images[camera] = _jpeg(frame_preprocessor.apply(raw, camera))
                policy.set_observation(
                    Observation(
                        observation_id=observation_id,
                        station_monotonic_ns=int((base_clock + history_time) * 1e9),
                        state=tuple(history_state.tolist()),
                        images=images,
                    )
                )
                observation_id += 1
                last_observation_time = history_time
            started = time.perf_counter()
            chunk = policy.get_chunk(tuple(state.tolist()))
            latencies.append(time.perf_counter() - started)
            remaining = max(1, round((end_s - query_time) * config.action_hz))
            execute = min(config.execution_horizon, len(chunk), remaining)
            for row_index in range(execute):
                next_time = min(end_s, query_time + (row_index + 1) / config.action_hz)
                state = chunk[row_index].copy()
                sim_times.append(next_time)
                simulated.append(state)
                if next_time >= end_s:
                    break
    finally:
        policy.close()

    sim_times_array = np.asarray(sim_times)
    simulated_array = np.asarray(simulated)
    reference = _interpolate_rows(dataset_times, dataset_states, sim_times_array)
    joint_error = simulated_array[:, JOINT_INDICES] - reference[:, JOINT_INDICES]
    per_joint = np.rad2deg(np.mean(np.abs(joint_error), axis=0))
    left_fk = PiperArmKinematics(lateral_offset_m=0.30)
    right_fk = PiperArmKinematics(lateral_offset_m=-0.30)
    tcp_errors = {"left": [], "right": []}
    for dream_state, recorded_state in zip(simulated_array, reference, strict=True):
        for side, kinematics, indices in (
            ("left", left_fk, slice(0, 6)),
            ("right", right_fk, slice(7, 13)),
        ):
            dream_tcp = kinematics.forward(dream_state[indices])[:3, 3]
            recorded_tcp = kinematics.forward(recorded_state[indices])[:3, 3]
            tcp_errors[side].append(float(np.linalg.norm(dream_tcp - recorded_tcp)))
    joints = simulated_array[:, JOINT_INDICES]
    report = {
        "dataset": dataset_id,
        "episode": episode,
        "task": task,
        "kind": "kinematic_shadow_rollout",
        "camera_source": "recorded demonstration",
        "state_source": "Dream actions from the first frame",
        "camera_preprocess_profile": frame_preprocessor.path,
        "start_s": start_s,
        "end_s": end_s,
        "dream_queries": len(query_times),
        "query_times_s": query_times,
        "inference_latency_s": {
            "mean": float(np.mean(latencies)),
            "max": float(np.max(latencies)),
            "over_live_timeout": int(
                np.count_nonzero(np.asarray(latencies) > config.action_timeout_s)
            ),
        },
        "trajectory": {
            "joint_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(joint_error**2)))),
            "joint_mae_deg": float(np.rad2deg(np.mean(np.abs(joint_error)))),
            "joint_mae_deg_by_name": dict(zip(JOINT_NAMES, per_joint.tolist(), strict=True)),
            "final_joint_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(joint_error[-1] ** 2)))
            ),
            "gripper_mae": float(
                np.mean(np.abs(simulated_array[:, GRIPPER_INDICES] - reference[:, GRIPPER_INDICES]))
            ),
            "left_tcp_rmse_cm": float(100 * np.sqrt(np.mean(np.square(tcp_errors["left"])))),
            "right_tcp_rmse_cm": float(
                100 * np.sqrt(np.mean(np.square(tcp_errors["right"])))
            ),
            "joint_limit_violation_count": int(
                np.count_nonzero((joints < JOINT_LOWER) | (joints > JOINT_UPPER))
            ),
        },
        "limitation": (
            "The cameras keep following the recorded demonstration after Dream diverges; "
            "this is kinematic shadow execution, not a closed-loop cloth simulation."
        ),
    }
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", *[f"dream_{i}" for i in range(14)], *[f"recorded_{i}" for i in range(14)]]
        )
        for timestamp, dream_state, recorded_state in zip(
            sim_times_array, simulated_array, reference, strict=True
        ):
            writer.writerow([timestamp, *dream_state.tolist(), *recorded_state.tolist()])
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        _render_shadow_video(
            readers=readers,
            paths=paths,
            sim_times=sim_times_array,
            simulated=simulated_array,
            reference=reference,
            output=output / "shadow_rollout.mp4",
            frame_preprocessor=frame_preprocessor,
        )
    finally:
        for reader in readers.values():
            reader.close()
    return report
