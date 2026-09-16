"""The PC-side agent: the only process allowed to command the Piper."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from robotics_stack.contracts import Observation, RobotSchema
from robotics_stack.control.streamer import MotionStreamer
from robotics_stack.control.supervisor import MotionSupervisor, RunState
from robotics_stack.hardware.base import RobotDriver
from robotics_stack.hardware.cameras import CameraHub
from robotics_stack.link.messages import observation_message, parse_action


@dataclass
class StationStatus:
    state: str
    session_id: str | None
    policy_connected: bool
    last_fault: str | None
    accepted_actions: int
    rejected_actions: int
    cameras: dict[str, dict[str, object]]


class RobotStation:
    def __init__(
        self,
        robot: RobotDriver,
        schema: RobotSchema,
        *,
        watchdog_ms: float = 400.0,
        action_age_ms: float = 250.0,
        motion_mode: str = "direct",
        stream_rate_hz: float = 100.0,
        cameras: CameraHub | None = None,
        home_action: tuple[float, ...] | None = None,
    ):
        self.robot = robot
        self.schema = schema
        self.supervisor = MotionSupervisor(schema, action_age_ms)
        self.watchdog_ms = watchdog_ms
        if motion_mode not in {"direct", "bounded"}:
            raise ValueError("motion_mode must be either 'direct' or 'bounded'")
        self.motion_mode = motion_mode
        self.stream_rate_hz = stream_rate_hz
        self.cameras = cameras
        self.home_action = home_action
        self.policy_connected = False
        self._observation_id = 0
        self._accepted_actions = 0
        self._rejected_actions = 0
        self._last_rejection: str | None = None
        self._connection: Any | None = None
        self._send_lock = asyncio.Lock()
        self._streamer: MotionStreamer | None = None

    def connect(self) -> None:
        self.robot.connect()
        try:
            if self.cameras is not None:
                self.cameras.connect()
            if self.motion_mode == "bounded":
                self._streamer = MotionStreamer(self.robot, self.schema, self.stream_rate_hz)
                self._streamer.start()
            self.supervisor.connected()
        except BaseException:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        try:
            if self._streamer is not None:
                self._streamer.hold()
        finally:
            if self._streamer is not None:
                self._streamer.stop()
                self._streamer = None
            if self.cameras is not None:
                self.cameras.disconnect()
            self.robot.disconnect()
            self.supervisor.state = RunState.DISCONNECTED

    def validate_hardware(self) -> None:
        if self.supervisor.state == RunState.DISCONNECTED:
            raise RuntimeError("connect the robot before validation")
        observation = self.robot.observation()
        self.schema.validate_action(observation)
        if self.cameras is not None:
            self.cameras.validate()

    def arm(self) -> str:
        self.validate_hardware()
        return self.supervisor.arm()

    def start(self) -> None:
        self.supervisor.start()

    def pause(self, reason: str = "operator pause") -> None:
        if self._streamer is not None:
            self._streamer.hold()
        else:
            self.robot.hold()
        self.supervisor.pause(reason)

    def home(self) -> None:
        self.pause("home requested")
        if self.home_action is None:
            raise RuntimeError("home is not configured for this Piper rig")
        if self._streamer is None:
            self.robot.set_target(self.home_action)
        else:
            self._streamer.set_target(self.home_action)

    def clear_fault(self) -> None:
        if self._streamer is not None:
            self._streamer.hold()
        else:
            self.robot.hold()
        self.supervisor.reset_fault()

    def status(self) -> StationStatus:
        health = self.cameras.health() if self.cameras is not None else {}
        return StationStatus(
            state=self.supervisor.state.value,
            session_id=self.supervisor.session_id,
            policy_connected=self.policy_connected,
            last_fault=self.supervisor.last_fault or self._last_rejection,
            accepted_actions=self._accepted_actions,
            rejected_actions=self._rejected_actions,
            cameras={
                name: {
                    "connected": camera.connected,
                    "age_ms": camera.age_ms,
                    "error": camera.error,
                }
                for name, camera in health.items()
            },
        )

    async def serve(self, host: str, port: int) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise RuntimeError("install the station extra to run the station") from exc
        async with serve(self._handle_policy, host, port, max_size=8 * 1024 * 1024):
            await asyncio.Future()

    async def _handle_policy(self, websocket: Any) -> None:
        if self._connection is not None:
            await websocket.close(code=4001, reason="one policy agent is allowed")
            return
        self._connection = websocket
        self.policy_connected = True
        try:
            hello = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5.0))
            compatible_schema = int(hello.get("schema_version", -1)) == self.schema.version
            if hello.get("type") != "hello" or not compatible_schema:
                raise ValueError("policy schema does not match station")
            await self._send(
                {
                    "type": "hello",
                    "schema": self.schema.to_dict(),
                    "state": self.supervisor.state.value,
                    "session_id": self.supervisor.session_id,
                }
            )
            observation_task = asyncio.create_task(self._observation_loop())
            receive_task = asyncio.create_task(self._receive_loop(websocket))
            _done, pending = await asyncio.wait(
                {observation_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:
            self.supervisor.fault(f"policy link failed: {exc}")
            self.robot.hold()
        finally:
            self.policy_connected = False
            self._connection = None
            if self.supervisor.state == RunState.RUNNING:
                self.pause("policy link disconnected")

    async def _observation_loop(self) -> None:
        while self._connection is not None:
            if self.supervisor.watchdog_expired(self.watchdog_ms):
                self.pause("policy action watchdog expired")
            if self.supervisor.state == RunState.RUNNING:
                self._observation_id += 1
                observation = Observation(
                    observation_id=self._observation_id,
                    station_monotonic_ns=time.monotonic_ns(),
                    state=self.robot.observation(),
                    images=self.cameras.frames() if self.cameras is not None else {},
                )
                message = observation_message(observation)
                message["session_id"] = self.supervisor.session_id
                await self._send(message)
            await asyncio.sleep(1.0 / self.schema.control_hz)

    async def _receive_loop(self, websocket: Any) -> None:
        async for raw_message in websocket:
            message = json.loads(raw_message)
            if message.get("type") != "action":
                continue
            action = parse_action(message)
            verdict = self.supervisor.validate(action)
            if verdict.accepted:
                if self._streamer is None:
                    self.robot.set_target(action.values)
                else:
                    self._streamer.set_target(action.values)
                self._accepted_actions += 1
            else:
                self._rejected_actions += 1
                self._last_rejection = verdict.reason

    async def _send(self, message: dict[str, Any]) -> None:
        if self._connection is None:
            return
        async with self._send_lock:
            await self._connection.send(json.dumps(message))
