"""Local datagram input for a VR retargeter or another operator device."""

from __future__ import annotations

import json
import socket
import threading

import numpy as np

from robotics_stack.simulation.controls import (
    SimulationCommand,
    SimulationCommandType,
    SimulationControls,
)


class OperatorTargetReceiver:
    """Accept finite joint targets from a local JSON/UDP producer.

    Each datagram is ``{"values": [..]}``. The receiver intentionally accepts
    only localhost by default: the real robot path must use its station-local
    authority and timestamp bridge instead of an unauthenticated network input.
    """

    def __init__(self, controls: SimulationControls, *, action_size: int, port: int):
        if action_size <= 0 or not 1 <= port <= 65535:
            raise ValueError("action_size and UDP port must be positive")
        self._controls = controls
        self._action_size = action_size
        self._port = port
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("127.0.0.1", self._port))
        self._socket.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="operator-target-receiver"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close()
        self._thread = None
        self._socket = None

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                raw, _address = self._socket.recvfrom(16 * 1024)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                value = json.loads(raw)
                values = tuple(float(item) for item in value["values"])
                if len(values) != self._action_size or not np.isfinite(values).all():
                    continue
                self._controls.submit(
                    SimulationCommand(SimulationCommandType.OPERATOR_TARGET, values)
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
                continue
