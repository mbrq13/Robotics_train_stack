"""Bind a VR teleoperator to the station's explicit DAgger authority handoff."""

from __future__ import annotations

from robotics_stack.services.station import RobotStation
from robotics_stack.teleoperators.guidance.tempo import FixedRateTargetBridge, TargetBridgeSettings
from robotics_stack.teleoperators.vr.session import VrTeleoperator
from robotics_stack.teleoperators.vr.tracking import TrackingFrame


class GuidedVrHandoff:
    """Run a correction epoch without bypassing station-local authority checks."""

    def __init__(
        self,
        station: RobotStation,
        teleoperator: VrTeleoperator,
        *,
        bridge_settings: TargetBridgeSettings | None = None,
    ) -> None:
        if station.schema != teleoperator.schema:
            raise ValueError("VR teleoperator schema does not match the station")
        self.station = station
        self.teleoperator = teleoperator
        self.bridge_settings = bridge_settings or TargetBridgeSettings()
        self._bridge: FixedRateTargetBridge | None = None
        self._submitted = False

    def begin_correction(
        self, frame: TrackingFrame, *, sides: tuple[str, ...] = ("left", "right")
    ) -> int:
        """Hold policy motion, synchronize measured state, then grant VR control."""
        generation = self.station.begin_operator_correction()
        self.teleoperator.set_state(self.station.robot.observation())
        self.teleoperator.engage(frame, sides=sides)
        self._bridge = self.station.create_operator_target_bridge(self.bridge_settings)
        self._bridge.start()
        self._submitted = False
        return generation

    def submit_tracking(self, frame: TrackingFrame) -> bool:
        """Publish the newest VR target into the station-local interpolation bridge."""
        if self._bridge is None:
            raise RuntimeError("begin a VR correction before submitting tracking")
        target = self.teleoperator.step(frame)
        if target is None:
            return False
        self._bridge.submit(
            target, timestamp_s=frame.timestamp_ns / 1_000_000_000, reset=not self._submitted
        )
        self._submitted = True
        return True

    def finish_correction(self) -> int:
        """Stop target playback and return the station to a held state."""
        if self._bridge is None:
            raise RuntimeError("no VR correction is active")
        try:
            self._bridge.stop()
        finally:
            self._bridge = None
            self.teleoperator.disengage()
        return self.station.finish_operator_control()

    def resume_policy(self) -> str:
        """Create the new policy session required after a human intervention."""
        return self.station.resume_policy()
