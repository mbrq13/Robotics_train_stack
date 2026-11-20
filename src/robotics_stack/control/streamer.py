"""Station-local trajectory streamer with velocity and acceleration bounds."""

from __future__ import annotations

import threading
import logging

import numpy as np

from robotics_stack.contracts import RobotSchema
from robotics_stack.robots.base import RobotDriver

logger = logging.getLogger(__name__)


class MotionStreamer:
    """Moves toward the latest safe target without making policy latency a CAN concern."""

    def __init__(self, robot: RobotDriver, schema: RobotSchema, rate_hz: float = 100.0):
        if rate_hz <= 0:
            raise ValueError("stream rate must be positive")
        self.robot = robot
        self.schema = schema
        self.rate_hz = rate_hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._velocity = np.zeros(schema.action_size, dtype=np.float64)
        self._lower = np.asarray([limit.minimum for limit in schema.joint_limits])
        self._upper = np.asarray([limit.maximum for limit in schema.joint_limits])

    def start(self) -> None:
        initial = np.asarray(self.robot.observation(), dtype=np.float64)
        self._current = initial.copy()
        self._target = initial.copy()
        self._thread = threading.Thread(target=self._run, daemon=True, name="motion-streamer")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_target(self, values: tuple[float, ...]) -> None:
        target = self.schema.validate_action(values)
        with self._lock:
            self._target = target

    def hold(self) -> None:
        """Stop trajectory generation at measured pose, or the last safe command.

        A feedback sample can be slightly beyond a configured hard stop after a
        physical overshoot. Do not turn that invalid sample into a new command:
        retain the last in-range command so the station can pause cleanly.
        """
        try:
            measured = self.schema.validate_action(self.robot.observation())
        except Exception as exc:
            logger.error("Cannot use out-of-range feedback for motion hold: %s", exc)
            with self._lock:
                if self._current is None:
                    raise
                measured = np.clip(self._current, self._lower, self._upper)
        with self._lock:
            self._current = measured.copy()
            self._target = measured.copy()
            self._velocity.fill(0.0)
        self.robot.set_target(tuple(float(value) for value in measured))

    def _run(self) -> None:
        interval = 1.0 / self.rate_hz
        while not self._stop.wait(interval):
            with self._lock:
                if self._current is None or self._target is None:
                    continue
                speed = np.asarray([limit.max_speed for limit in self.schema.joint_limits])
                acceleration = np.asarray(
                    [limit.max_acceleration for limit in self.schema.joint_limits]
                )
                requested_velocity = np.clip(
                    (self._target - self._current) / interval, -speed, speed
                )
                velocity_step = np.clip(
                    requested_velocity - self._velocity,
                    -acceleration * interval,
                    acceleration * interval,
                )
                self._velocity = np.clip(self._velocity + velocity_step, -speed, speed)
                step = self._velocity * interval
                remaining = self._target - self._current
                reached = np.abs(step) >= np.abs(remaining)
                candidate = self._current + np.where(reached, remaining, step)
                # Acceleration-limited interpolation can otherwise overshoot a
                # stop after an abrupt target replacement. Keep the internal
                # state valid as well as the public target contract.
                self._current = np.clip(candidate, self._lower, self._upper)
                clipped = self._current != candidate
                self._velocity[reached] = 0.0
                self._velocity[clipped] = 0.0
                command = tuple(float(value) for value in self._current)
            try:
                self.robot.set_target(command)
            except Exception:
                # A driver failure must not leave the streaming thread dead
                # while the station is trying to hold a valid last command.
                logger.exception("Motion streamer command rejected; retaining last safe target")
                with self._lock:
                    self._target = self._current.copy()
                    self._velocity.fill(0.0)
