"""Optional Viser panel for a kinematic guided-collection replay."""

from __future__ import annotations

from typing import Any

import numpy as np

from robotics_stack.contracts import RobotSchema
from robotics_stack.simulation.controls import (
    SimulationCommand,
    SimulationCommandType,
    SimulationControls,
)


class ViserGuidancePanel:
    """Visualizes joint targets and exposes deliberate simulation controls.

    It does not load a robot mesh. This keeps the simulator independent of
    unreviewed assets while still showing the commanded bimanual configuration.
    """

    def __init__(self, schema: RobotSchema, controls: SimulationControls, *, port: int):
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError("install the sim extra to use the Viser panel") from exc
        self._schema = schema
        self._controls = controls
        self._server = viser.ViserServer(port=port)
        self._operator_target = np.zeros(schema.action_size, dtype=np.float64)
        self._phase = self._server.gui.add_text(
            "control phase", initial_value="policy_run", disabled=True
        )
        self._add_button("Pause policy", SimulationCommandType.PAUSE)
        self._add_button("Take recovery control", SimulationCommandType.TAKEOVER)
        self._add_button("Take correction control", SimulationCommandType.CORRECTION)
        self._add_button("Resume policy", SimulationCommandType.RESUME)
        self._add_button("Save episode", SimulationCommandType.SAVE)
        self._add_button("Stop simulation", SimulationCommandType.STOP)
        joints = zip(schema.action_names, schema.joint_limits, strict=True)
        for index, (name, limit) in enumerate(joints):
            slider = self._server.gui.add_slider(
                name,
                min=limit.minimum,
                max=limit.maximum,
                step=(limit.maximum - limit.minimum) / 500.0,
                initial_value=0.0,
            )

            @slider.on_update
            def _set_target(_event: Any, joint: int = index, handle: Any = slider) -> None:
                self._operator_target[joint] = float(handle.value)
                self._controls.submit(
                    SimulationCommand(
                        SimulationCommandType.OPERATOR_TARGET,
                        tuple(float(value) for value in self._operator_target),
                    )
                )

    def _add_button(self, label: str, kind: SimulationCommandType) -> None:
        button = self._server.gui.add_button(label)

        @button.on_click
        def _submit(_event: Any) -> None:
            self._controls.submit(SimulationCommand(kind))

    def publish(self, state: tuple[float, ...], *, phase: str) -> None:
        self._phase.value = phase
        self._operator_target[:] = np.asarray(state, dtype=np.float64)
