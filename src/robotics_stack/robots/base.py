"""Hardware boundary. Implementations expose physical state, never policy tensors."""

from __future__ import annotations

from typing import Protocol

from robotics_stack.contracts import RobotSchema


class RobotDriver(Protocol):
    schema: RobotSchema

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def observation(self) -> tuple[float, ...]: ...

    def set_target(self, action: tuple[float, ...]) -> None: ...

    def hold(self) -> None: ...

    def home(self) -> None: ...
