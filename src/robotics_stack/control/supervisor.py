"""Robot-side motion authority. This module never imports a policy or UI."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass

from robotics_stack.contracts import ContractError, PolicyAction, RobotSchema
from robotics_stack.guidance.authority import ControlAuthority, ControlSource


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

    def __init__(
        self,
        schema: RobotSchema,
        max_action_age_ms: float = 250.0,
        max_scheduled_action_age_ms: float | None = None,
    ):
        self.schema = schema
        if max_action_age_ms <= 0:
            raise ValueError("max_action_age_ms must be positive")
        scheduled_age_ms = (
            max_action_age_ms
            if max_scheduled_action_age_ms is None
            else max_scheduled_action_age_ms
        )
        if scheduled_age_ms <= 0:
            raise ValueError("max_scheduled_action_age_ms must be positive")
        self.max_action_age_ns = int(max_action_age_ms * 1_000_000)
        self.max_scheduled_action_age_ns = int(scheduled_age_ms * 1_000_000)
        self.state = RunState.DISCONNECTED
        self.authority = ControlAuthority()
        self.session_id: str | None = None
        self.last_sequence_id = -1
        self.last_fault: str | None = None
        self.last_action_at_ns: int | None = None
        self.run_started_at_ns: int | None = None

    def connected(self) -> None:
        self.state = RunState.READY
        self.last_fault = None

    def arm(self) -> str:
        if self.state not in {RunState.READY, RunState.PAUSED}:
            raise RuntimeError(f"cannot arm while {self.state.value}")
        self.session_id = uuid.uuid4().hex
        self.last_sequence_id = -1
        self.last_action_at_ns = None
        self.run_started_at_ns = None
        self.state = RunState.ARMED
        return self.session_id

    def start(self) -> None:
        if self.state != RunState.ARMED:
            raise RuntimeError("robot must be armed before execution")
        self.state = RunState.RUNNING
        self.authority.activate_policy()
        self.run_started_at_ns = time.monotonic_ns()

    def pause(self, reason: str = "operator pause") -> None:
        if self.state != RunState.DISCONNECTED:
            self.state = RunState.PAUSED
        if self.authority.snapshot.source is not ControlSource.HOLD:
            self.authority.hold()
        self.last_fault = reason

    def fault(self, reason: str) -> None:
        self.state = RunState.FAULT
        self.authority.stop()
        self.last_fault = reason
        self.session_id = None

    def reset_fault(self) -> None:
        if self.state != RunState.FAULT:
            raise RuntimeError("no fault to reset")
        self.state = RunState.READY
        self.last_fault = None
        self.last_sequence_id = -1

    @property
    def control_generation(self) -> int:
        return self.authority.snapshot.generation

    def validate(self, action: PolicyAction, now_ns: int | None = None) -> GuardResult:
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.state != RunState.RUNNING:
            return GuardResult(False, f"station is {self.state.value}")
        if action.session_id != self.session_id:
            return GuardResult(False, "session does not match current armed session")
        if not self.authority.accepts(ControlSource.POLICY, action.control_generation):
            return GuardResult(False, "action belongs to an inactive control generation")
        if action.schema_version != self.schema.version:
            return GuardResult(False, "schema version does not match robot")
        if action.sequence_id <= self.last_sequence_id:
            return GuardResult(False, "action sequence is stale or duplicated")
        maximum_age_ns = (
            self.max_scheduled_action_age_ns if action.scheduled else self.max_action_age_ns
        )
        if now - action.station_monotonic_ns > maximum_age_ns:
            return GuardResult(False, "action was computed from an expired observation")
        try:
            self.schema.validate_action(action.values)
        except ContractError as exc:
            return GuardResult(False, str(exc))
        self.last_sequence_id = action.sequence_id
        self.last_action_at_ns = now
        return GuardResult(True, "accepted")

    def watchdog_expired(
        self,
        timeout_ms: float,
        *,
        first_action_timeout_ms: float | None = None,
        now_ns: int | None = None,
    ) -> bool:
        if self.state != RunState.RUNNING:
            return False
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.last_action_at_ns is None:
            if self.run_started_at_ns is None or first_action_timeout_ms is None:
                return False
            return now - self.run_started_at_ns > int(first_action_timeout_ms * 1_000_000)
        return now - self.last_action_at_ns > int(timeout_ms * 1_000_000)
