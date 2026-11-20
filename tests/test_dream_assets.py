from __future__ import annotations

import json
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from robotics_stack.contracts import ContractError
from robotics_stack.policies.dream1 import validate_arm_file

ASSETS = Path(__file__).parents[1] / "assets" / "robots" / "piper"


def test_all_referenced_meshes_are_complete_binary_stl_files() -> None:
    report = validate_arm_file(ASSETS / "piper_arm.json", require_confirmation=False)
    assert report["mesh_count"] == 10
    root = ET.parse(ASSETS / "piper_arm.urdf").getroot()
    names = {mesh.attrib["filename"] for mesh in root.findall(".//mesh")}
    assert names == {f"meshes/{path.name}" for path in (ASSETS / "meshes").glob("*.STL")}
    for name in names:
        data = (ASSETS / name).read_bytes()
        triangles = struct.unpack_from("<I", data, 80)[0]
        assert triangles > 0
        assert len(data) == 84 + 50 * triangles
        for triangle in struct.iter_unpack("<12fH", data[84:]):
            assert all(math.isfinite(value) for value in triangle[:12])


def test_missing_mesh_refuses_preflight_even_without_tcp_confirmation(tmp_path: Path) -> None:
    shutil.copytree(ASSETS, tmp_path / "piper")
    (tmp_path / "piper" / "meshes" / "link7.STL").unlink()
    with pytest.raises(ContractError, match="missing or empty Piper mesh"):
        validate_arm_file(tmp_path / "piper" / "piper_arm.json", require_confirmation=False)


def test_jaw_axis_matches_the_finger_slide_in_flange_coordinates() -> None:
    root = ET.parse(ASSETS / "piper_arm.urdf").getroot()
    finger = root.find("joint[@name='finger_left']")
    roll, pitch, yaw = map(float, finger.find("origin").attrib["rpy"].split())
    assert pitch == yaw == 0
    # The finger's local Z slide rotates about flange X by pi/2.
    slide = np.array([0, -math.sin(roll), math.cos(roll)])
    arm = json.loads((ASSETS / "piper_arm.json").read_text())
    assert abs(np.dot(slide, arm["jaw"])) > 0.99999
    assert abs(np.dot(arm["approach"], arm["jaw"])) < 1e-8


def test_official_client_extracts_only_the_six_arm_joints() -> None:
    arm_file = pytest.importorskip("dream.arm_file")
    document = json.loads((ASSETS / "piper_arm.json").read_text())
    chain, joints, _limits, speeds = arm_file._chain(
        (ASSETS / "piper_arm.urdf").read_bytes(),
        document["base"], document["flange"], ASSETS / "piper_arm.json",
    )
    assert joints == tuple(f"joint{i}" for i in range(1, 7))
    assert speeds == (0.5,) * 6
    assert ET.fromstring(chain).findall(".//mesh") == []


def test_full_urdf_loads_visual_and_collision_meshes() -> None:
    yourdfpy = pytest.importorskip("yourdfpy")
    robot = yourdfpy.URDF.load(
        str(ASSETS / "piper_arm.urdf"), build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    assert len(robot.scene.geometry) == 10
    assert len(robot.collision_scene.geometry) == 10
