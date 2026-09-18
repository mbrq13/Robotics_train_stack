import numpy as np

from robotics_stack.contracts import JointLimit, RobotSchema
from robotics_stack.guidance.authority import GuidancePhase
from robotics_stack.simulation.dagger import DatasetReplay, GuidedSimulation, ReplayFrame


class FixedPolicy:
    def predict(self, _observation):
        return (0.2, -0.2)


def _schema() -> RobotSchema:
    return RobotSchema(
        name="test",
        action_names=("a", "b"),
        state_names=("a", "b"),
        joint_limits=(JointLimit(-1, 1, 1, 1), JointLimit(-1, 1, 1, 1)),
        camera_names=("front",),
    )


def test_simulation_records_policy_and_operator_segments(tmp_path) -> None:
    simulation = GuidedSimulation(
        _schema(),
        DatasetReplay([ReplayFrame({"front": b"frame"}, 0)]),
        FixedPolicy(),
        task="test",
    )

    simulation.start_policy()
    policy = simulation.step_policy()
    simulation.pause()
    simulation.take_control(correction=True)
    operator = simulation.step_operator((0.4, -0.4))
    simulation.resume_policy()
    resumed = simulation.step_policy()

    assert policy.phase is GuidancePhase.POLICY_RUN
    assert operator.phase is GuidancePhase.CORRECTION
    assert resumed.phase is GuidancePhase.POLICY_RUN
    assert simulation.state == (0.2, -0.2)
    exported = simulation.episode.export_npz(tmp_path / "guided.npz")
    recorded = np.load(exported)
    assert recorded["operator_mask"].tolist() == [False, True, False]
    assert recorded["images_front_offsets"].tolist() == [0, 5, 10, 15]
