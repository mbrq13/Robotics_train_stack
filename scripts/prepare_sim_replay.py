"""Export a bounded camera replay archive from a LeRobot dataset.

The output is consumed by ``rstack simulate``. It contains the RGB frames and
recorded states needed to exercise a policy and guided-control transitions;
it does not change the source dataset.
"""

from __future__ import annotations

import argparse

import numpy as np


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image(value) -> np.ndarray:
    image = _array(value)
    if image.ndim != 3:
        raise ValueError(f"expected a three-dimensional image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB image data, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255)
    return image.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LeRobot frames for the kinematic simulator"
    )
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repository identifier")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cameras", nargs="+", required=True, help="camera names without prefix")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    if args.start < 0 or args.frames <= 0:
        raise ValueError("--start must be non-negative and --frames must be positive")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("install the policy-lerobot extra to read a LeRobot dataset") from exc

    dataset = LeRobotDataset(args.dataset)
    stop = min(len(dataset), args.start + args.frames)
    if args.start >= stop:
        raise ValueError("requested replay range is outside the dataset")
    states: list[np.ndarray] = []
    images = {name: [] for name in args.cameras}
    for index in range(args.start, stop):
        sample = dataset[index]
        states.append(_array(sample["observation.state"]).astype(np.float32).reshape(-1))
        for name in args.cameras:
            images[name].append(_image(sample[f"observation.images.{name}"]))
    np.savez_compressed(
        args.output,
        states=np.stack(states),
        **{f"images_{name}": np.stack(values) for name, values in images.items()},
    )


if __name__ == "__main__":
    main()
