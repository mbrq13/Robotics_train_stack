"""Wire messages. JSON is intentionally explicit and independently versioned."""

from __future__ import annotations

import base64
from typing import Any

from robotics_stack.contracts import Observation, PolicyAction

PROTOCOL_VERSION = 1


def observation_message(observation: Observation) -> dict[str, Any]:
    return {
        "type": "observation",
        "protocol_version": PROTOCOL_VERSION,
        "observation_id": observation.observation_id,
        "station_monotonic_ns": observation.station_monotonic_ns,
        "control_generation": observation.control_generation,
        "state": list(observation.state),
        "images": {
            name: base64.b64encode(image).decode("ascii")
            for name, image in observation.images.items()
        },
    }


def parse_observation(value: dict[str, Any]) -> Observation:
    if value.get("type") != "observation" or value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported observation message")
    return Observation(
        observation_id=int(value["observation_id"]),
        station_monotonic_ns=int(value["station_monotonic_ns"]),
        state=tuple(float(number) for number in value["state"]),
        images={name: base64.b64decode(image) for name, image in value.get("images", {}).items()},
        control_generation=int(value.get("control_generation", 0)),
    )


def action_message(action: PolicyAction) -> dict[str, Any]:
    return {
        "type": "action",
        "protocol_version": PROTOCOL_VERSION,
        "session_id": action.session_id,
        "sequence_id": action.sequence_id,
        "observation_id": action.observation_id,
        "station_monotonic_ns": action.station_monotonic_ns,
        "schema_version": action.schema_version,
        "values": list(action.values),
        "scheduled": action.scheduled,
        "control_generation": action.control_generation,
    }


def parse_action(value: dict[str, Any]) -> PolicyAction:
    if value.get("type") != "action" or value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported action message")
    return PolicyAction(
        session_id=str(value["session_id"]),
        sequence_id=int(value["sequence_id"]),
        observation_id=int(value["observation_id"]),
        station_monotonic_ns=int(value["station_monotonic_ns"]),
        schema_version=int(value["schema_version"]),
        values=tuple(float(number) for number in value["values"]),
        scheduled=bool(value.get("scheduled", False)),
        control_generation=int(value.get("control_generation", 0)),
    )
