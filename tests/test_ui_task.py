from robotics_stack.contracts import JointLimit, RobotSchema
from robotics_stack.robots.fake import FakeRobot
from robotics_stack.services.station import RobotStation
from robotics_stack.ui.app import create_app


def test_prompt_api_validates_and_only_changes_while_stopped():
    from fastapi.testclient import TestClient

    schema = RobotSchema(
        name="test",
        action_names=("joint",),
        state_names=("joint",),
        joint_limits=(JointLimit(-1, 1, 1, 1),),
    )
    station = RobotStation(FakeRobot(schema), schema)
    station.connect()
    try:
        with TestClient(create_app(station)) as client:
            assert (
                client.post("/api/task", json={"task": "  grapes to basket  "}).status_code == 200
            )
            assert client.get("/api/status").json()["task"] == "grapes to basket"
            assert station._observation_message()["task"] == "grapes to basket"
            for bad in ("", " ", 123, "x" * 2001):
                assert client.post("/api/task", json={"task": bad}).status_code == 409
            station.arm()
            assert client.post("/api/task", json={"task": "new"}).status_code == 409
            station.start()
            assert client.post("/api/task", json={"task": "new"}).status_code == 409
            station.pause()
            assert client.post("/api/task", json={"task": "new"}).status_code == 200
            assert 'id="task"' in client.get("/").text
    finally:
        station.disconnect()
