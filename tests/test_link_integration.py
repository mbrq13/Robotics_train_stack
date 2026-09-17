from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import pytest

from robotics_stack.contracts import JointLimit, PolicyAction, RobotSchema
from robotics_stack.hardware.fake import FakeRobot
from robotics_stack.link.messages import action_message, parse_observation
from robotics_stack.runtime.station import RobotStation

websockets = pytest.importorskip("websockets")


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _schema() -> RobotSchema:
    return RobotSchema(
        name="link-test",
        action_names=("joint", "gripper"),
        state_names=("joint", "gripper"),
        joint_limits=(JointLimit(-1, 1, 1, 1), JointLimit(0, 1, 1, 1)),
        control_hz=100,
    )


def test_station_accepts_only_current_network_action() -> None:
    async def scenario() -> None:
        schema = _schema()
        station = RobotStation(FakeRobot(schema), schema)
        station.connect()
        port = _port()
        server = asyncio.create_task(station.serve("127.0.0.1", port))
        await asyncio.sleep(0.05)
        try:
            from websockets.asyncio.client import connect

            async with connect(f"ws://127.0.0.1:{port}") as client:
                await client.send(json.dumps({"type": "hello", "schema_version": 1}))
                hello = json.loads(await client.recv())
                assert hello["state"] == "ready"
                session = station.arm()
                station.start()
                observation_wire = json.loads(await client.recv())
                observation = parse_observation(observation_wire)
                assert observation.control_generation == 0
                await client.send(
                    json.dumps(
                        action_message(
                            PolicyAction(
                                session_id=session,
                                sequence_id=1,
                                observation_id=observation.observation_id,
                                station_monotonic_ns=observation.station_monotonic_ns,
                                schema_version=1,
                                values=(0.2, 0.5),
                                control_generation=observation.control_generation,
                            )
                        )
                    )
                )
                await asyncio.sleep(0.03)
                assert station.status().accepted_actions == 1
        finally:
            server.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server
            station.disconnect()

    asyncio.run(scenario())
