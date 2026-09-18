"""Runnable policy replay with pause, operator takeover, and episode export."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from robotics_stack.simulation.controls import (
    SimulationCommand,
    SimulationCommandType,
    SimulationControls,
)
from robotics_stack.simulation.dagger import GuidedSimulation
from robotics_stack.teleoperators.guidance.authority import ControlSource


def start_terminal_controls(controls: SimulationControls) -> None:
    """Read line-based controls without blocking the fixed-rate simulation loop."""

    mapping = {
        "p": SimulationCommandType.PAUSE,
        "t": SimulationCommandType.TAKEOVER,
        "c": SimulationCommandType.CORRECTION,
        "r": SimulationCommandType.RESUME,
        "s": SimulationCommandType.SAVE,
        "q": SimulationCommandType.STOP,
    }

    def read() -> None:
        while True:
            try:
                value = input().strip().lower()
            except EOFError:
                return
            if value in mapping:
                controls.submit(SimulationCommand(mapping[value]))
            if value == "q":
                return

    threading.Thread(target=read, daemon=True, name="simulation-terminal-controls").start()


def run_guided_simulation(
    simulation: GuidedSimulation,
    controls: SimulationControls,
    *,
    output: str | Path,
    max_steps: int = 0,
    panel: object | None = None,
) -> Path:
    """Run until stopped and export a provenance-preserving NPZ episode."""
    simulation.start_policy()
    latest_operator_target: tuple[float, ...] | None = None
    period_s = 1.0 / simulation.schema.control_hz
    steps = 0
    stopped = False
    while not stopped and (max_steps <= 0 or steps < max_steps):
        started_at = time.perf_counter()
        save = False
        for command in controls.drain():
            try:
                if (
                    command.kind is SimulationCommandType.PAUSE
                    and simulation.phase.value == "policy_run"
                ):
                    simulation.pause()
                elif command.kind is SimulationCommandType.TAKEOVER:
                    simulation.take_control()
                elif command.kind is SimulationCommandType.CORRECTION:
                    simulation.take_control(correction=True)
                elif command.kind is SimulationCommandType.RESUME:
                    simulation.resume_policy()
                elif command.kind is SimulationCommandType.OPERATOR_TARGET:
                    latest_operator_target = command.values
                elif command.kind is SimulationCommandType.SAVE:
                    save = True
                elif command.kind is SimulationCommandType.STOP:
                    stopped = True
            except RuntimeError:
                continue
        if simulation.authority.snapshot.source is ControlSource.POLICY:
            simulation.step_policy()
            steps += 1
        elif (
            simulation.authority.snapshot.source is ControlSource.OPERATOR
            and latest_operator_target
        ):
            simulation.step_operator(latest_operator_target)
            steps += 1
        if panel is not None:
            panel.publish(simulation.state, phase=simulation.phase.value)
        if save:
            return simulation.episode.export_npz(output)
        time.sleep(max(0.0, period_s - (time.perf_counter() - started_at)))
    return simulation.episode.export_npz(output)
