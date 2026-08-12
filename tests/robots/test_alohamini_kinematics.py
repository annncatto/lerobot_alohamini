from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lerobot.robots.alohamini.kinematics import AlohaMiniJointMapping, AlohaMiniKinematics
from scripts.sync_alohamini_kinematics_assets import DEFAULT_OUTPUT_ROOT

REFERENCE_Q = np.array([0.0, -1.571, 1.571, 0.0, 0.0, 0.0])


def test_generated_kinematics_asset_manifest_is_complete():
    manifest = json.loads((DEFAULT_OUTPUT_ROOT / "kinematics.json").read_text())
    assert manifest["asset"] == "alohamini2pro"
    assert manifest["source_files"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["source_files"].values())
    assert {
        "alohamini2pro_kinematic.urdf",
        "alohamini2pro_left_kinematic.urdf",
        "alohamini2pro_right_kinematic.urdf",
        "alohamini2pro_moveit.urdf",
        "alohamini2pro.srdf",
    } <= {path.name for path in DEFAULT_OUTPUT_ROOT.iterdir()}


def test_generated_arm_urdfs_have_symmetric_joint_trees_and_tcp():
    signatures = {}
    for side in ("left", "right"):
        path = DEFAULT_OUTPUT_ROOT / f"alohamini2pro_{side}_kinematic.urdf"
        root = ET.parse(path).getroot()
        joints = root.findall("joint")
        links = root.findall("link")
        child_links = {joint.find("child").get("link") for joint in joints}
        assert {link.get("name") for link in links} - child_links == {f"{side}_Base"}
        assert root.find(f"joint[@name='{side}_tcp_joint']") is not None
        signatures[side] = [
            (
                joint.get("name").removeprefix(f"{side}_"),
                dict(joint.find("origin").attrib),
                {} if joint.find("axis") is None else dict(joint.find("axis").attrib),
            )
            for joint in joints
            if joint.get("name") != f"{side}_tcp_joint"
        ]
    assert signatures["left"] == signatures["right"]


def test_moveit_asset_keeps_collision_geometry_and_semantic_groups():
    urdf = ET.parse(DEFAULT_OUTPUT_ROOT / "alohamini2pro_moveit.urdf").getroot()
    assert urdf.get("name") == "alohamini2pro_moveit"
    assert urdf.find("link[@name='root']/inertial") is None
    assert urdf.find("material[@name='white']") is not None
    meshes = urdf.findall(".//mesh")
    assert meshes
    assert all(
        mesh.get("filename", "").startswith("package://alohamini_description/alohamini2pro/")
        for mesh in meshes
    )
    assert urdf.find("joint[@name='left_tcp_joint']") is not None
    assert urdf.find("joint[@name='right_tcp_joint']") is not None
    collision_meshes = urdf.findall(".//collision/geometry/mesh")
    assert collision_meshes
    assert all(
        "collision_meshes/" in mesh.get("filename", "")
        or mesh.get("filename", "").endswith("/base_link.STL")
        for mesh in collision_meshes
    )
    assert urdf.findall(".//collision/geometry/box")

    srdf = ET.parse(DEFAULT_OUTPUT_ROOT / "alohamini2pro.srdf").getroot()
    assert srdf.get("name") == urdf.get("name")
    groups = {group.get("name") for group in srdf.findall("group")}
    assert {"left_arm", "right_arm", "dual_arms", "left_gripper", "right_gripper"} <= groups


def test_single_measured_calibration_is_identical_for_both_logical_sides():
    mapping = AlohaMiniJointMapping()
    assert mapping.manifest["shared_calibration"]["left_alias"]["inherits"] == (
        "hardware_joint_map_right.yaml"
    )
    for joint in (*mapping.joint_order, "gripper"):
        calibration = mapping.calibration(joint)
        assert mapping.urdf_joint_name("left", joint).startswith("left_")
        assert mapping.urdf_joint_name("right", joint).startswith("right_")
        assert mapping.raw_tick_to_urdf(joint, calibration.reference_tick) == pytest.approx(
            calibration.reference_q_rad
        )


def test_encoder_and_urdf_mapping_round_trip_within_one_tick():
    mapping = AlohaMiniJointMapping()
    one_tick_rad = 2.0 * np.pi / mapping.ticks_per_revolution
    for joint in (*mapping.joint_order, "gripper"):
        calibration = mapping.calibration(joint)
        samples = [
            calibration.lower_rad,
            calibration.reference_q_rad,
            calibration.upper_rad,
        ]
        for q_rad in samples:
            tick = mapping.urdf_to_raw_tick(joint, q_rad)
            recovered = mapping.raw_tick_to_urdf(joint, tick, near_q_rad=q_rad)
            assert abs(recovered - q_rad) <= one_tick_rad * calibration.joint_per_encoder_ratio


def test_shoulder_pan_positive_urdf_direction_decreases_raw_ticks():
    mapping = AlohaMiniJointMapping()
    calibration = mapping.calibration("shoulder_pan")
    one_tick_rad = 2.0 * np.pi / mapping.ticks_per_revolution

    assert calibration.sign == -1
    assert mapping.urdf_to_raw_tick("shoulder_pan", one_tick_rad) == (
        calibration.reference_tick - 1
    ) % mapping.ticks_per_revolution
    assert calibration.lower_rad == pytest.approx(-2.133767275948927)
    assert calibration.upper_rad == pytest.approx(2.2396119503130363)


@pytest.mark.parametrize("drive_mode", [0, 1])
def test_lerobot_normalization_round_trip(drive_mode):
    mapping = AlohaMiniJointMapping()
    for normalization, values in (
        ("range_m100_100", (-100.0, -20.0, 0.0, 75.0, 100.0)),
        ("range_0_100", (0.0, 20.0, 50.0, 75.0, 100.0)),
    ):
        for value in values:
            tick = mapping.lerobot_to_raw_tick(
                value,
                range_min=500,
                range_max=3500,
                drive_mode=drive_mode,
                normalization=normalization,
            )
            recovered = mapping.raw_tick_to_lerobot(
                tick,
                range_min=500,
                range_max=3500,
                drive_mode=drive_mode,
                normalization=normalization,
            )
            assert recovered == pytest.approx(value, abs=0.07)


def test_left_and_right_local_forward_kinematics_are_identical():
    left = AlohaMiniKinematics("left")
    right = AlohaMiniKinematics("right")
    samples = (
        REFERENCE_Q,
        REFERENCE_Q + np.array([0.2, 0.1, -0.15, 0.1, -0.1, 0.3]),
    )
    for q_rad in samples:
        assert np.allclose(left.forward_kinematics(q_rad), right.forward_kinematics(q_rad))


def test_forward_kinematics_matches_optional_placo_backend():
    pytest.importorskip("placo")
    from lerobot.model.kinematics import RobotKinematics

    samples = (
        REFERENCE_Q,
        REFERENCE_Q + np.array([0.2, 0.1, -0.15, 0.1, -0.1, 0.3]),
    )
    for side in ("left", "right"):
        core = AlohaMiniKinematics(side)
        joint_names = [core.mapping.urdf_joint_name(side, joint) for joint in core.joint_order]
        placo = RobotKinematics(
            str(core.asset_dir / f"alohamini2pro_{side}_kinematic.urdf"),
            f"{side}_tcp",
            joint_names,
        )
        for q_rad in samples:
            expected = core.forward_kinematics(q_rad)
            actual = placo.forward_kinematics(np.rad2deg(q_rad))
            assert np.allclose(actual, expected, atol=2e-10)


@pytest.mark.parametrize("side", ["left", "right"])
def test_inverse_kinematics_recovers_reachable_tcp_pose(side):
    kinematics = AlohaMiniKinematics(side)
    goal_q = REFERENCE_Q + np.array([0.1, 0.05, -0.05, 0.02, 0.03, -0.02])
    target = kinematics.forward_kinematics(goal_q)

    result = kinematics.inverse_kinematics(target, REFERENCE_Q)

    assert result.success, result
    assert result.reason == "converged"
    assert result.position_error_m < 1e-4
    assert result.orientation_error_rad < 1e-3
    assert np.all(result.q_rad >= kinematics.lower_limits)
    assert np.all(result.q_rad <= kinematics.upper_limits)


def test_inverse_kinematics_rejects_invalid_and_unreachable_targets():
    kinematics = AlohaMiniKinematics("right")
    invalid = np.eye(4)
    invalid[0, 0] = np.nan
    assert kinematics.inverse_kinematics(invalid, REFERENCE_Q).reason == "invalid_target"

    invalid_rotation = np.eye(4)
    invalid_rotation[0, 0] = 2.0
    assert kinematics.inverse_kinematics(invalid_rotation, REFERENCE_Q).reason == "invalid_rotation"

    unreachable = np.eye(4)
    unreachable[:3, 3] = [10.0, 10.0, 10.0]
    result = kinematics.inverse_kinematics(unreachable, REFERENCE_Q, max_iterations=20)
    assert not result.success
    assert result.reason == "max_iterations"
