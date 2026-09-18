"""Robot-driver registry kept separate from station and policy runtime code."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from robotics_stack.contracts import RobotSchema
from robotics_stack.robots.base import RobotDriver

RobotFactory = Callable[[RobotSchema, Mapping[str, Any]], RobotDriver]
_DRIVERS: dict[str, RobotFactory] = {}


@dataclass(frozen=True)
class RobotProfile:
    """Declarative selection of one hardware adapter for a station."""

    driver: str
    options: dict[str, Any]
    action_space: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobotProfile:
        driver = str(value.get("driver", "")).strip()
        if not driver:
            raise ValueError("robot.driver is required")
        options = value.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("robot.options must be a mapping")
        action_space = str(value.get("action_space", "")).strip()
        if not action_space:
            raise ValueError("robot.action_space is required")
        return cls(driver=driver, options=dict(options), action_space=action_space)


def register_robot_driver(name: str, factory: RobotFactory) -> None:
    """Register a hardware adapter without importing it into the runtime core."""
    if not name:
        raise ValueError("robot driver name must not be empty")
    if name in _DRIVERS:
        raise ValueError(f"robot driver is already registered: {name}")
    _DRIVERS[name] = factory


def available_robot_drivers() -> tuple[str, ...]:
    return tuple(sorted(_DRIVERS))


def create_robot_driver(schema: RobotSchema, profile: RobotProfile) -> RobotDriver:
    """Build the selected adapter after checking its declared action space."""
    if schema.action_space != profile.action_space:
        raise ValueError(
            "schema action_space does not match robot profile: "
            f"schema={schema.action_space!r}, profile={profile.action_space!r}"
        )
    if profile.driver not in _DRIVERS:
        # Built-in adapters register themselves lazily. Third-party adapters
        # can register before this call without forcing SDK imports elsewhere.
        from robotics_stack.robots import piper  # noqa: F401

    try:
        factory = _DRIVERS[profile.driver]
    except KeyError as exc:
        options = ", ".join(available_robot_drivers()) or "none installed"
        raise ValueError(f"unknown robot driver {profile.driver!r}; available: {options}") from exc
    return factory(schema, profile.options)
