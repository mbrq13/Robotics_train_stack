"""Explicit ownership of robot motion during guided collection."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ControlSource(str, enum.Enum):  # noqa: UP042 - keep the module importable on older station hosts
    HOLD = "hold"
    POLICY = "policy"
    OPERATOR = "operator"


class GuidancePhase(str, enum.Enum):  # noqa: UP042 - keep the module importable on older station hosts
    INACTIVE = "inactive"
    POLICY_RUN = "policy_run"
    HOLDING = "holding"
    RECOVERY = "recovery"
    CORRECTION = "correction"


@dataclass(frozen=True)
class AuthoritySnapshot:
    phase: GuidancePhase
    source: ControlSource
    generation: int


class ControlAuthority:
    """Small state machine that makes the active action source unambiguous.

    A generation advances at every completed handoff after the initial policy run.
    Transport layers attach it to actions, so an action produced before a handoff
    cannot take control after the handoff has completed.
    """

    def __init__(self) -> None:
        self._snapshot = AuthoritySnapshot(
            phase=GuidancePhase.INACTIVE,
            source=ControlSource.HOLD,
            generation=0,
        )

    @property
    def snapshot(self) -> AuthoritySnapshot:
        return self._snapshot

    def activate_policy(self) -> AuthoritySnapshot:
        if self._snapshot.phase not in {GuidancePhase.INACTIVE, GuidancePhase.HOLDING}:
            raise RuntimeError("policy can only start from an inactive or holding state")
        return self._move(GuidancePhase.POLICY_RUN, ControlSource.POLICY, advance=False)

    def hold(self) -> AuthoritySnapshot:
        if self._snapshot.phase not in {
            GuidancePhase.POLICY_RUN,
            GuidancePhase.RECOVERY,
            GuidancePhase.CORRECTION,
        }:
            raise RuntimeError("hold requires an active controller")
        return self._move(GuidancePhase.HOLDING, ControlSource.HOLD)

    def begin_recovery(self) -> AuthoritySnapshot:
        if self._snapshot.phase != GuidancePhase.HOLDING:
            raise RuntimeError("recovery requires the robot to be holding")
        return self._move(GuidancePhase.RECOVERY, ControlSource.OPERATOR)

    def begin_correction(self) -> AuthoritySnapshot:
        if self._snapshot.phase not in {GuidancePhase.HOLDING, GuidancePhase.RECOVERY}:
            raise RuntimeError("correction requires a hold or recovery phase")
        return self._move(GuidancePhase.CORRECTION, ControlSource.OPERATOR)

    def finish_operator_control(self) -> AuthoritySnapshot:
        if self._snapshot.phase not in {GuidancePhase.RECOVERY, GuidancePhase.CORRECTION}:
            raise RuntimeError("operator control is not active")
        return self._move(GuidancePhase.HOLDING, ControlSource.HOLD)

    def stop(self) -> AuthoritySnapshot:
        if self._snapshot.phase == GuidancePhase.INACTIVE:
            return self._snapshot
        return self._move(GuidancePhase.INACTIVE, ControlSource.HOLD)

    def accepts(self, source: ControlSource, generation: int) -> bool:
        return self._snapshot.source is source and self._snapshot.generation == generation

    def _move(
        self,
        phase: GuidancePhase,
        source: ControlSource,
        *,
        advance: bool = True,
    ) -> AuthoritySnapshot:
        generation = self._snapshot.generation + int(advance)
        self._snapshot = AuthoritySnapshot(phase=phase, source=source, generation=generation)
        return self._snapshot
