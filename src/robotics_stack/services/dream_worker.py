"""Chunked Dream-1 worker connected to the hardware station."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from typing import Any

from robotics_stack.contracts import ContractError, PolicyAction, RobotSchema
from robotics_stack.policies.dream1 import Dream1Config, Dream1Policy, validate_dream_schema
from robotics_stack.transport.messages import action_message, parse_observation

# The official physical loop requires another start after a long cold-pool wait.
MOVE_AFTER_WAIT_LIMIT_S = 60.0


def _retryable_error(exc: Exception) -> bool:
    from dream.errors import (
        DreamAuthError,
        DreamConfigError,
        DreamError,
        DreamObservationError,
        DreamServerError,
    )

    if isinstance(exc, (DreamAuthError, DreamConfigError, DreamObservationError)):
        return False
    if isinstance(exc, DreamServerError):
        return exc.retryable
    return isinstance(exc, (DreamError, ContractError, TimeoutError))


def _replacement_start(executed_during_request: int, elapsed_s: float, action_hz: float) -> int:
    """Rows already traversed in both command time and wall time."""
    due_by_time = max(0, round(elapsed_s * action_hz))
    return min(max(0, executed_during_request), due_by_time)


async def run_dream_agent(config: Dream1Config, *, policy: Dream1Policy | None = None) -> None:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise RuntimeError("install the dream optional dependency to run the Dream worker") from exc

    owned_policy = policy is None
    policy = Dream1Policy(config) if policy is None else policy
    try:
        async with connect(config.station_url, max_size=8 * 1024 * 1024) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "hello",
                        "schema_version": 1,
                        "mode": "observe" if config.dry_run else "control",
                    }
                )
            )
            hello = json.loads(await websocket.recv())
            schema = RobotSchema.from_dict(hello["schema"])
            validate_dream_schema(schema, config)
            await _run_loop(websocket, policy, schema, dry_run=config.dry_run)
    finally:
        if owned_policy:
            policy.close()


async def _run_loop(
    websocket: Any,
    policy: Dream1Policy,
    schema: RobotSchema,
    *,
    dry_run: bool,
) -> None:
    queue: deque[tuple[float, ...]] = deque()
    latest: tuple[Any, str] | None = None
    epoch: tuple[str, int] | None = None
    chunk_source: tuple[int, int, int] | None = None
    chunk_executed = 0
    executed_total = 0
    sequence_id = 0
    inference: asyncio.Task[Any] | None = None
    stopped = asyncio.Event()
    send_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    last_error: str | None = None
    underruns = 0
    last_emitted: tuple[float, ...] | None = None
    first_chunk_installed = False
    episode_started_at: float | None = None
    fatal_error: Exception | None = None
    reset_before_refill = False

    async def send(message: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send(json.dumps(message))

    async def receive() -> None:
        nonlocal chunk_executed, chunk_source, episode_started_at, epoch
        nonlocal first_chunk_installed, inference, last_error, latest
        nonlocal last_emitted, executed_total
        nonlocal reset_before_refill
        try:
            while not stopped.is_set():
                if dry_run:
                    await send({"type": "observation_request"})
                raw = await websocket.recv()
                message = json.loads(raw)
                if message.get("type") != "observation":
                    continue
                observation = parse_observation(message)
                session_id = str(message.get("session_id") or "dry-run")
                candidate_epoch = (session_id, observation.control_generation)
                async with state_lock:
                    if epoch != candidate_epoch:
                        epoch = candidate_epoch
                        queue.clear()
                        chunk_source = None
                        chunk_executed = 0
                        executed_total = 0
                        last_emitted = None
                        reset_before_refill = False
                        first_chunk_installed = False
                        episode_started_at = time.perf_counter()
                        await asyncio.to_thread(policy.reset_episode)
                    latest = (observation, session_id)
                    policy.set_observation(observation)
                    can_infer = inference is None or inference.done()
                    needs_chunk = can_infer and (
                        not queue or policy.refill_due(chunk_executed, len(queue))
                    )
                    if needs_chunk:
                        inference = asyncio.create_task(
                            produce(
                                candidate_epoch,
                                observation,
                                executed_total,
                                last_emitted,
                                time.perf_counter(),
                            )
                        )
                if dry_run:
                    await asyncio.sleep(1.0 / schema.control_hz)
        finally:
            stopped.set()

    async def produce(
        start_epoch: tuple[str, int],
        observation: Any,
        executed_at_request: int,
        commanded: tuple[float, ...] | None,
        started_at: float,
    ) -> None:
        nonlocal chunk_executed, chunk_source, first_chunk_installed, last_error
        nonlocal fatal_error
        nonlocal reset_before_refill
        try:
            if reset_before_refill:
                await asyncio.to_thread(policy.reset_episode)
                if epoch != start_epoch:
                    return
                policy.set_observation(observation)
                reset_before_refill = False
            chunk = await asyncio.to_thread(policy.get_chunk, commanded)
            landed_at = time.perf_counter()
            elapsed_s = landed_at - started_at
            if elapsed_s > policy.config.action_timeout_s:
                raise TimeoutError(
                    f"Dream chunk took {elapsed_s:.2f}s; limit is "
                    f"{policy.config.action_timeout_s:.2f}s"
                )
            actions = [
                tuple(float(value) for value in schema.validate_action(row)) for row in chunk
            ]
            async with state_lock:
                if epoch != start_epoch:
                    return
                if (
                    not dry_run
                    and not first_chunk_installed
                    and episode_started_at is not None
                    and landed_at - episode_started_at > MOVE_AFTER_WAIT_LIMIT_S
                ):
                    raise RuntimeError(
                        "Dream is ready after a wait longer than 60s; no motion was sent. "
                        "Restart the worker, then validate, arm and start again."
                    )
                # Match the official physical loop: the replacement chunk starts
                # at the row reached while its request was in flight. Counting
                # executed rows avoids jumping through a period where the arm held.
                executed_during_request = max(0, executed_total - executed_at_request)
                skipped = _replacement_start(
                    executed_during_request, elapsed_s, schema.control_hz
                )
                if skipped >= len(actions):
                    raise TimeoutError("Dream chunk landed after its complete horizon")
                queue.clear()
                queue.extend(actions[skipped:])
                chunk_executed = 0
                chunk_source = (
                    observation.observation_id,
                    observation.station_monotonic_ns,
                    observation.control_generation,
                )
                first_chunk_installed = True
                last_error = None
        except Exception as exc:  # The station watchdog remains the motion authority.
            if epoch != start_epoch:
                return
            if not _retryable_error(exc):
                queue.clear()
                fatal_error = exc
                return
            last_error = str(exc)
            from dream.errors import DreamActionError, DreamServerError

            reset_before_refill = (
                exc.reason not in ("rate_limited", "queue_timeout")
                if isinstance(exc, DreamServerError)
                else (
                    not isinstance(exc, (DreamActionError, ContractError))
                    and type(exc) is not TimeoutError
                )
            )
            retry_after = getattr(exc, "retry_after_s", None)
            delay_s = (
                float(retry_after)
                if isinstance(retry_after, int | float)
                and not isinstance(retry_after, bool)
                and math.isfinite(retry_after)
                and retry_after > 0
                else 1.0
            )
            await asyncio.sleep(delay_s)

    async def emit() -> None:
        nonlocal chunk_executed, executed_total, last_emitted, sequence_id, underruns
        period = 1.0 / schema.control_hz
        deadline = time.perf_counter()
        last_status = 0.0
        while not stopped.is_set():
            if fatal_error is not None:
                raise fatal_error
            deadline += period
            async with state_lock:
                if queue and epoch is not None and chunk_source is not None:
                    values = queue.popleft()
                    chunk_executed += 1
                    executed_total += 1
                    last_emitted = values
                    if not dry_run:
                        sequence_id += 1
                        observation_id, timestamp_ns, generation = chunk_source
                        await send(
                            action_message(
                                PolicyAction(
                                    session_id=epoch[0],
                                    sequence_id=sequence_id,
                                    observation_id=observation_id,
                                    station_monotonic_ns=timestamp_ns,
                                    schema_version=schema.version,
                                    values=values,
                                    scheduled=True,
                                    control_generation=generation,
                                )
                            )
                        )
                elif latest is not None:
                    underruns += 1
                    if (
                        not first_chunk_installed
                        and episode_started_at is not None
                        and time.perf_counter() - episode_started_at
                        > policy.config.first_chunk_timeout_s
                    ):
                        raise TimeoutError(
                            "Dream produced no first chunk within "
                            f"{policy.config.first_chunk_timeout_s:.0f}s"
                        )
            now = time.perf_counter()
            if now - last_status >= 1.0:
                last_status = now
                timing = policy.timing
                await send(
                    {
                        "type": "rtc_status",
                        "enabled": True,
                        "queue_depth": len(queue),
                        "underruns": underruns,
                        "error": last_error,
                        "p95_ms": timing.get("rtt_ms"),
                        "max_ms": timing.get("rtt_ms"),
                        "samples": sequence_id,
                    }
                )
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            # Do not burst through missed command ticks after an event-loop stall.
            deadline = max(deadline, time.perf_counter())

    receiver = asyncio.create_task(receive())
    actor = asyncio.create_task(emit())
    try:
        done, _pending = await asyncio.wait(
            {receiver, actor}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            task.result()
    finally:
        stopped.set()
        receiver.cancel()
        actor.cancel()
        if inference is not None:
            inference.cancel()
        pending = (receiver, actor, inference) if inference is not None else (receiver, actor)
        await asyncio.gather(*pending, return_exceptions=True)
