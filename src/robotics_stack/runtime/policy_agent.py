"""Thor-side policy worker. It cannot command hardware directly."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from robotics_stack.contracts import PolicyAction, RobotSchema
from robotics_stack.link.messages import action_message, parse_observation
from robotics_stack.policy.mlp import StateMlpPolicy


async def run_policy_agent(url: str, checkpoint: str | Path) -> None:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise RuntimeError("install the policy extra to run a policy agent") from exc
    policy, manifest = StateMlpPolicy.load(checkpoint)
    async with connect(url, max_size=8 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps({"type": "hello", "schema_version": manifest.schema.version}))
        hello = json.loads(await websocket.recv())
        schema = RobotSchema.from_dict(hello["schema"])
        if schema != manifest.schema:
            raise RuntimeError("checkpoint schema is not compatible with the connected station")
        sequence_id = 0
        async for raw_message in websocket:
            message = json.loads(raw_message)
            if message.get("type") != "observation":
                continue
            observation = parse_observation(message)
            session_id = message.get("session_id")
            if not session_id:
                continue
            sequence_id += 1
            action = policy.predict(observation.state)
            await websocket.send(
                json.dumps(
                    action_message(
                        PolicyAction(
                            session_id=session_id,
                            sequence_id=sequence_id,
                            observation_id=observation.observation_id,
                            station_monotonic_ns=observation.station_monotonic_ns,
                            schema_version=schema.version,
                            values=action,
                        )
                    )
                )
            )
