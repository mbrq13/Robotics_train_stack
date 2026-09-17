"""Policy inference worker."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from robotics_stack.contracts import Observation, PolicyAction, RobotSchema
from robotics_stack.link.messages import action_message, parse_observation
from robotics_stack.policy.checkpoints import DeployedPolicy, load_deployed_policy
from robotics_stack.policy.rtc import LatencyTracker, RtcActionQueue, RtcSettings


async def _run_standard_agent(websocket: Any, policy: DeployedPolicy, schema: RobotSchema) -> None:
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
        action = await asyncio.to_thread(policy.predict, observation)
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


async def _run_rtc_agent(
    websocket: Any,
    policy: DeployedPolicy,
    schema: RobotSchema,
    settings: RtcSettings,
) -> None:
    """Run inference and action emission independently at the station cadence."""
    await asyncio.to_thread(policy.configure_rtc, settings)
    queue = RtcActionQueue(settings, schema.action_size)
    latency = LatencyTracker()
    latest: tuple[Observation, str] | None = None
    active_session: str | None = None
    observation_ready = asyncio.Event()
    stopped = asyncio.Event()
    send_lock = asyncio.Lock()
    sequence_id = 0

    async def send(message: dict[str, object]) -> None:
        async with send_lock:
            await websocket.send(json.dumps(message))

    async def receive_observations() -> None:
        nonlocal active_session, latest, latency
        try:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                if message.get("type") != "observation":
                    continue
                session_id = str(message.get("session_id", ""))
                if not session_id:
                    continue
                if active_session != session_id:
                    active_session = session_id
                    queue.clear()
                    latency = LatencyTracker()
                latest = (parse_observation(message), session_id)
                observation_ready.set()
        finally:
            stopped.set()
            observation_ready.set()

    async def produce_chunks() -> None:
        while not stopped.is_set():
            await observation_ready.wait()
            observation_ready.clear()
            if stopped.is_set() or latest is None or len(queue) > settings.refill_threshold:
                continue
            observation, session_at_start = latest
            previous_raw = queue.raw_left_over()
            expected_delay = latency.delay_steps(settings.control_hz)
            started_at = time.perf_counter()
            try:
                chunk = await asyncio.to_thread(
                    policy.predict_rtc_chunk,
                    observation,
                    inference_delay_steps=expected_delay,
                    previous_raw_actions=previous_raw,
                )
                elapsed_s = time.perf_counter() - started_at
                measured_delay = math.ceil(elapsed_s * settings.control_hz)
                latency.add(elapsed_s)
                # Ignore work produced for a replaced session.
                if active_session != session_at_start:
                    continue
                queue.merge(
                    chunk,
                    inference_delay_steps=measured_delay,
                    observation_id=observation.observation_id,
                    station_monotonic_ns=observation.station_monotonic_ns,
                )
            except Exception as exc:
                await send({"type": "rtc_status", "error": str(exc), **latency.snapshot()})
                await asyncio.sleep(settings.action_period_s)

    async def emit_actions() -> None:
        nonlocal sequence_id
        next_tick = time.perf_counter()
        last_status_at = 0.0
        while not stopped.is_set():
            next_tick += settings.action_period_s
            if active_session is not None and len(queue):
                queued = queue.pop()
                if queued is not None:
                    sequence_id += 1
                    await send(
                        action_message(
                            PolicyAction(
                                session_id=active_session,
                                sequence_id=sequence_id,
                                observation_id=queued.observation_id,
                                station_monotonic_ns=queued.station_monotonic_ns,
                                schema_version=schema.version,
                                values=tuple(float(value) for value in queued.action),
                                scheduled=True,
                            )
                        )
                    )
            now = time.perf_counter()
            if now - last_status_at >= 1.0:
                last_status_at = now
                await send(
                    {
                        "type": "rtc_status",
                        "queue_depth": len(queue),
                        "underruns": queue.underruns,
                        "enabled": True,
                        **latency.snapshot(),
                    }
                )
            await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))

    receiver = asyncio.create_task(receive_observations())
    producer = asyncio.create_task(produce_chunks())
    actor = asyncio.create_task(emit_actions())
    try:
        await receiver
    finally:
        stopped.set()
        observation_ready.set()
        for task in (producer, actor):
            task.cancel()
        await asyncio.gather(producer, actor, return_exceptions=True)


async def run_policy_agent(
    url: str,
    checkpoint: str | Path,
    *,
    task: str = "",
    device: str = "cuda",
    schema_version: int = 1,
    state_names: tuple[str, ...] = (),
    execution_mode: str = "standard",
    rtc_execution_horizon: int | None = None,
    rtc_refill_threshold: int | None = None,
) -> None:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise RuntimeError("install the policy extra to run a policy agent") from exc
    if execution_mode not in {"standard", "rtc"}:
        raise ValueError("execution_mode must be 'standard' or 'rtc'")
    policy = load_deployed_policy(
        checkpoint, task=task, device=device, state_names=state_names
    )
    async with connect(url, max_size=8 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps({"type": "hello", "schema_version": schema_version}))
        hello = json.loads(await websocket.recv())
        schema = RobotSchema.from_dict(hello["schema"])
        if schema.version != schema_version:
            raise RuntimeError("station schema version changed during handshake")
        policy.descriptor.validate_station(schema)
        if execution_mode == "standard":
            await _run_standard_agent(websocket, policy, schema)
            return
        training_delay = policy.descriptor.rtc_training_max_delay
        horizon = rtc_execution_horizon or training_delay
        settings = RtcSettings(
            control_hz=schema.control_hz,
            training_max_delay=training_delay,
            execution_horizon=horizon,
            refill_threshold=rtc_refill_threshold or max(training_delay, horizon),
        )
        await _run_rtc_agent(websocket, policy, schema, settings)
