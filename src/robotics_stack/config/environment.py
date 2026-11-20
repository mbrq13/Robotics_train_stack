"""Explicit deployment overrides: process environment, then local .env, then YAML."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


def dream_overrides() -> dict:
    values = {**dotenv_values(Path.cwd() / ".env"), **os.environ}
    result = {}
    for variable, field in {
        "RSTACK_DREAM_MODEL": "checkpoint",
        "RSTACK_DREAM_SOLVER": "ik_solve",
        "RSTACK_DREAM_TASK": "task",
        "RSTACK_STATION_URL": "station_url",
    }.items():
        if values.get(variable):
            result[field] = values[variable]
    dry = values.get("RSTACK_DREAM_DRY_RUN")
    if dry:
        if dry.lower() not in {"true", "false"}:
            raise ValueError("RSTACK_DREAM_DRY_RUN must be true or false")
        result["dry_run"] = dry.lower() == "true"
    return result
