import socket
import time

from robotics_stack.simulation.controls import SimulationCommandType, SimulationControls
from robotics_stack.simulation.operator_input import OperatorTargetReceiver


def test_receiver_accepts_only_a_complete_finite_target() -> None:
    controls = SimulationControls()
    receiver = OperatorTargetReceiver(controls, action_size=2, port=18976)
    receiver.start()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(b'{"values": [0.1, -0.2]}', ("127.0.0.1", 18976))
        deadline = time.monotonic() + 1.0
        commands = ()
        while time.monotonic() < deadline and not commands:
            commands = controls.drain()
            time.sleep(0.01)
        assert len(commands) == 1
        assert commands[0].kind is SimulationCommandType.OPERATOR_TARGET
        assert commands[0].values == (0.1, -0.2)
    finally:
        receiver.stop()
