"""Robot-side motion authority. This module never imports a policy or UI."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass

from robotics_stack.contracts import ContractError, PolicyAction, RobotSchema


class RunState(str, enum.Enum):  # noqa: UP042 - keep the runtime importable on commissioning PCs
    DISCONNECTED = "disconnected"
    READY = "ready"
    ARMED = "armed"
    RUNNING = "running"
    PAUSED = "paused"
    FAULT = "fault"


@dataclass(frozen=True)
class GuardResult:
    accepted: bool
    reason: str


class MotionSupervisor:
    """Validates remote actions and owns the explicit human arming lifecycle."""

    def __init__(self, schema: RobotSchema, max_action_age_ms: float = 250.0):
        self.schema = schema
        self.max_action_age_ns = int(max_action_age_ms * 1_000_000)
        self.state = RunState.DISCONNECTED
        self.session_id: str | None = None
        self.last_sequence_id = -1
        self.last_fault: str | None = None
        self.last_action_at_ns: int | None = None

    def connected(self) -> None:
        self.state = RunState.READY
        self.last_fault = None

    def arm(self) -> str:
        if self.state not in {RunState.READY, RunState.PAUSED}:
            raise RuntimeError(f"cannot arm while {self.state.value}")
        self.session_id = uuid.uuid4().hex
        self.last_sequence_id = -1
        self.state = RunState.ARMED
        return self.session_id

    def start(self) -> None:
        if self.state != RunState.ARMED:
            raise RuntimeError("robot must be armed before execution")
        self.state = RunState.RUNNING

    def pause(self, reason: str = "operator pause") -> None:
        if self.state != RunState.DISCONNECTED:
            self.state = RunState.PAUSED
        self.last_fault = reason

    def fault(self, reason: str) -> None:
        self.state = RunState.FAULT
        self.last_fault = reason
        self.session_id = None

    def reset_fault(self) -> None:
        if self.state != RunState.FAULT:
            raise RuntimeError("no fault to reset")
        self.state = RunState.READY
        self.last_fault = None
        self.last_sequence_id = -1

    def validate(self, action: PolicyAction, now_ns: int | None = None) -> GuardResult:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.state != RunState.RUNNING:
            return GuardResult(False, f"station is {self.state.value}")
        if action.session_id != self.session_id:
            return GuardResult(False, "session does not match current armed session")
        if action.schema_version != self.schema.version:
            return GuardResult(False, "schema version does not match robot")
        if action.sequence_id <= self.last_sequence_id:
            return GuardResult(False, "action sequence is stale or duplicated")
        if now - action.station_monotonic_ns > self.max_action_age_ns:
            return GuardResult(False, "action was computed from an expired observation")
        try:
            self.schema.validate_action(action.values)
        except ContractError as exc:
            return GuardResult(False, str(exc))
        self.last_sequence_id = action.sequence_id
        self.last_action_at_ns = now
        return GuardResult(True, "accepted")

    def watchdog_expired(self, timeout_ms: float, now_ns: int | None = None) -> bool:
        if self.state != RunState.RUNNING or self.last_action_at_ns is None:
            return False
        now = time.monotonic_ns() if now_ns is None else now_ns
        return now - self.last_action_at_ns > int(timeout_ms * 1_000_000)
