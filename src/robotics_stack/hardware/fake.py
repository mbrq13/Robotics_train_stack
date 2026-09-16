"""Deterministic robot implementation used for integration and dry-run tests."""

from __future__ import annotations

import threading

import numpy as np

from robotics_stack.contracts import RobotSchema


class FakeRobot:
    def __init__(self, schema: RobotSchema):
        self.schema = schema
        self._state = np.zeros(schema.action_size, dtype=np.float64)
        self._connected = False
        self._lock = threading.Lock()

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def observation(self) -> tuple[float, ...]:
        if not self._connected:
            raise RuntimeError("fake robot is disconnected")
        with self._lock:
            return tuple(float(value) for value in self._state)

    def set_target(self, action: tuple[float, ...]) -> None:
        if not self._connected:
            raise RuntimeError("fake robot is disconnected")
        values = self.schema.validate_action(action)
        with self._lock:
            self._state = values

    def hold(self) -> None:
        self.observation()

    def home(self) -> None:
        if not self._connected:
            raise RuntimeError("fake robot is disconnected")
        with self._lock:
            self._state.fill(0.0)
