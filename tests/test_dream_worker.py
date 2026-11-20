import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_stack.config.loaders import load_station_config
from robotics_stack.contracts import Observation
from robotics_stack.services.dream_worker import _replacement_start, _retryable_error, _run_loop
from robotics_stack.transport.messages import observation_message


def test_chunk_replacement_skips_only_rows_executed_and_due() -> None:
    assert _replacement_start(executed_during_request=7, elapsed_s=0.25, action_hz=20) == 5
    assert _replacement_start(executed_during_request=3, elapsed_s=0.50, action_hz=20) == 3


def test_chunk_replacement_does_not_jump_through_a_hold() -> None:
    assert _replacement_start(executed_during_request=0, elapsed_s=2.0, action_hz=20) == 0


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.actions = asyncio.Queue()
        self.receiver_cancelled = False

    async def recv(self):
        try:
            return await self.incoming.get()
        except asyncio.CancelledError:
            self.receiver_cancelled = True
            raise

    async def send(self, raw):
        message = json.loads(raw)
        if message["type"] == "action":
            await self.actions.put(message)


class Policy:
    config = SimpleNamespace(action_timeout_s=3, first_chunk_timeout_s=900)
    timing = {}

    def __init__(self, row, error=None):
        self.row = row
        self.error = error
        self.anchors = []
        self.resets = 0

    def set_observation(self, observation):
        pass

    def reset_episode(self):
        self.resets += 1

    def refill_due(self, executed, remaining):
        return False

    def get_chunk(self, commanded):
        self.anchors.append(commanded)
        if self.error is not None:
            raise self.error
        return [self.row] * 48


def _schema():
    root = Path(__file__).parents[1]
    return load_station_config(root / "configs/robots/piper_bimanual_dream1.yaml")[0]


async def _observe(socket, row, generation):
    message = observation_message(Observation(generation, 1_000_000_000, row,
                                            control_generation=generation))
    message["session_id"] = f"session-{generation}"
    await socket.incoming.put(json.dumps(message))


def test_session_reset_discards_the_previous_command_anchor_and_cleans_up():
    async def scenario():
        schema = _schema()
        row = tuple((limit.minimum + limit.maximum) / 2 for limit in schema.joint_limits)
        socket = Socket()
        policy = Policy(row)
        worker = asyncio.create_task(_run_loop(socket, policy, schema, dry_run=False))
        try:
            await _observe(socket, row, 1)
            assert (await asyncio.wait_for(socket.actions.get(), 2))["control_generation"] == 1
            await _observe(socket, row, 2)
            while True:
                action = await asyncio.wait_for(socket.actions.get(), 2)
                if action["control_generation"] == 2:
                    break
            assert policy.anchors == [None, None]
            assert policy.resets == 2
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        assert socket.receiver_cancelled

    asyncio.run(scenario())


def test_fatal_api_error_exits_worker_instead_of_retrying():
    errors = pytest.importorskip("dream.errors")

    async def scenario():
        schema = _schema()
        row = tuple((limit.minimum + limit.maximum) / 2 for limit in schema.joint_limits)
        socket = Socket()
        policy = Policy(row, errors.DreamAuthError("invalid key"))
        await _observe(socket, row, 1)
        with pytest.raises(errors.DreamAuthError, match="invalid key"):
            await asyncio.wait_for(_run_loop(socket, policy, schema, dry_run=False), 2)
        assert socket.actions.empty()
        assert len(policy.anchors) == 1
        assert socket.receiver_cancelled

    asyncio.run(scenario())


def test_error_retry_classification_follows_server_refusals():
    errors = pytest.importorskip("dream.errors")
    assert _retryable_error(errors.DreamServerError("warming", retryable=True))
    assert _retryable_error(errors.DreamTimeout("timeout"))
    assert not _retryable_error(errors.DreamServerError("no entitlement", retryable=False))
    assert not _retryable_error(errors.DreamConfigError("bad config"))
    assert not _retryable_error(RuntimeError("programming failure"))


def test_cold_pool_reply_requires_another_start_before_motion(monkeypatch):
    pytest.importorskip("dream.errors")
    monkeypatch.setattr("robotics_stack.services.dream_worker.MOVE_AFTER_WAIT_LIMIT_S", 0)

    async def scenario():
        schema = _schema()
        row = tuple((limit.minimum + limit.maximum) / 2 for limit in schema.joint_limits)
        socket = Socket()
        policy = Policy(row)
        await _observe(socket, row, 1)
        with pytest.raises(RuntimeError, match="no motion was sent"):
            await asyncio.wait_for(_run_loop(socket, policy, schema, dry_run=False), 2)
        assert socket.actions.empty()
        assert socket.receiver_cancelled

    asyncio.run(scenario())


def test_ik_rounding_does_not_admit_real_limit_violations():
    from pathlib import Path

    import pytest

    from robotics_stack.config.loaders import load_station_config
    from robotics_stack.contracts import ContractError
    from robotics_stack.services.dream_worker import _validate_ik_action

    root = Path(__file__).parents[1]
    schema, _, _ = load_station_config(root / "configs/robots/piper_bimanual_dream1.yaml")
    row = [0.0] * 14
    row[0] = -2.618
    assert _validate_ik_action(schema, row)[0] == schema.joint_limits[0].minimum
    row[0] = -2.62
    with pytest.raises(ContractError):
        _validate_ik_action(schema, row)


def test_station_task_reaches_inference_and_resets_old_chunk():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Config:
        task: str = "yaml task"
        action_timeout_s: float = 3
        first_chunk_timeout_s: float = 900

    async def scenario():
        schema = _schema()
        row = tuple((limit.minimum + limit.maximum) / 2 for limit in schema.joint_limits)
        policy = Policy(row)
        policy.config = Config()
        observed_tasks = []
        policy.set_observation = lambda observation: observed_tasks.append(policy.config.task)
        socket = Socket()
        worker = asyncio.create_task(_run_loop(socket, policy, schema, dry_run=False))
        try:
            for generation, task in ((1, "Put grapes in yellow basket"), (2, "Pick up the ball")):
                msg = observation_message(Observation(generation, 1_000_000_000, row,
                                                      control_generation=generation))
                msg.update(session_id=f"session-{generation}", task=task)
                await socket.incoming.put(json.dumps(msg))
                while True:
                    action = await asyncio.wait_for(socket.actions.get(), 2)
                    if action["control_generation"] == generation:
                        break
                assert observed_tasks[-1] == task
            assert policy.resets == 2
            assert policy.anchors == [None, None]
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())
