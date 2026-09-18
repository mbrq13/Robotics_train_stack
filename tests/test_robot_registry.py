from __future__ import annotations

from robotics_stack.contracts import JointLimit, RobotSchema
from robotics_stack.robots.registry import (
    RobotProfile,
    create_robot_driver,
    register_robot_driver,
)


def _schema() -> RobotSchema:
    return RobotSchema(
        name="test-arm",
        action_names=("joint",),
        state_names=("joint",),
        joint_limits=(JointLimit(-1, 1, 1, 1),),
        action_space="radians",
    )


def test_registry_builds_a_driver_without_station_knowing_its_type() -> None:
    name = "test_registry_driver"
    received: dict[str, object] = {}

    def factory(schema: RobotSchema, options: dict[str, object]) -> object:
        received["schema"] = schema
        received["options"] = options
        return object()

    register_robot_driver(name, factory)  # type: ignore[arg-type]
    result = create_robot_driver(
        _schema(), RobotProfile(driver=name, action_space="radians", options={"port": "test"})
    )

    assert result is not None
    assert received["schema"].name == "test-arm"  # type: ignore[union-attr]
    assert received["options"] == {"port": "test"}
