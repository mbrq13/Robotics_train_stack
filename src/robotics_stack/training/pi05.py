"""Validated launcher for LeRobot Pi0.5 training configurations."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def validate_pi05_rtc_config(path: str | Path) -> dict:
    """Reject an incoherent RTC training config before allocating a GPU."""
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    policy = config.get("policy", {})
    if policy.get("type") != "pi05":
        raise ValueError("Pi0.5 training config requires policy.type: pi05")
    chunk_size = int(policy.get("chunk_size", 0))
    delay = int(policy.get("rtc_training_max_delay", 0))
    if chunk_size <= 0:
        raise ValueError("policy.chunk_size must be positive")
    if delay <= 0:
        raise ValueError("RTC training requires policy.rtc_training_max_delay > 0")
    if delay > chunk_size // 2:
        raise ValueError("rtc_training_max_delay must be <= half of chunk_size")
    return config


def pi05_train_command(path: str | Path) -> list[str]:
    """Build the public LeRobot command; training remains owned by LeRobot."""
    validate_pi05_rtc_config(path)
    return ["lerobot-train", "--config_path", str(Path(path))]


def run_pi05_training(path: str | Path, *, dry_run: bool = False) -> list[str]:
    command = pi05_train_command(path)
    if not dry_run:
        subprocess.run(command, check=True)
    return command
