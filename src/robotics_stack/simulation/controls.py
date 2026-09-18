"""Thread-safe commands accepted by the simulated guided-collection session."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from enum import Enum


class SimulationCommandType(str, Enum):  # noqa: UP042 - supports older station hosts
    PAUSE = "pause"
    TAKEOVER = "takeover"
    CORRECTION = "correction"
    RESUME = "resume"
    OPERATOR_TARGET = "operator_target"
    SAVE = "save"
    STOP = "stop"


@dataclass(frozen=True)
class SimulationCommand:
    kind: SimulationCommandType
    values: tuple[float, ...] = ()


class SimulationControls:
    """A bounded, UI-neutral command mailbox for simulation control."""

    def __init__(self) -> None:
        self._commands: queue.Queue[SimulationCommand] = queue.Queue(maxsize=32)

    def submit(self, command: SimulationCommand) -> None:
        try:
            self._commands.put_nowait(command)
        except queue.Full as exc:
            raise RuntimeError("simulation control queue is full") from exc

    def drain(self) -> tuple[SimulationCommand, ...]:
        commands: list[SimulationCommand] = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return tuple(commands)
