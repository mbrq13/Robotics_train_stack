import time

import numpy as np

from robotics_stack.control.streamer import MotionStreamer
from robotics_stack.contracts import JointLimit, RobotSchema
from robotics_stack.robots.fake import FakeRobot


def _schema() -> RobotSchema:
    return RobotSchema(
        name="streamer-test",
        action_names=("joint",),
        state_names=("joint",),
        joint_limits=(JointLimit(-1.0, 1.0, 1.0, 1.0),),
    )


def test_streamer_clamps_internal_interpolation_before_sending() -> None:
    robot = FakeRobot(_schema())
    robot.connect()
    streamer = MotionStreamer(robot, _schema(), rate_hz=500)
    streamer.start()
    try:
        # Simulate an internal overshoot caused by a replaced target. The next
        # CAN command must be limited rather than killing the worker thread.
        with streamer._lock:
            streamer._current = np.array([1.1])
            streamer._target = np.array([1.0])
        time.sleep(0.03)
        assert robot.observation() == (1.0,)
        assert streamer._thread is not None and streamer._thread.is_alive()
    finally:
        streamer.stop()
        robot.disconnect()


def test_streamer_hold_retains_last_safe_target_when_feedback_is_invalid() -> None:
    class FeedbackFailureRobot(FakeRobot):
        fail_feedback = False

        def observation(self):
            if self.fail_feedback:
                raise ValueError("joint feedback is outside configured limits")
            return super().observation()

    robot = FeedbackFailureRobot(_schema())
    robot.connect()
    robot.set_target((0.4,))
    streamer = MotionStreamer(robot, _schema())
    streamer.start()
    try:
        robot.fail_feedback = True
        streamer.hold()
        assert tuple(streamer._current) == (0.4,)
        assert tuple(streamer._target) == (0.4,)
    finally:
        streamer.stop()
        robot.disconnect()
