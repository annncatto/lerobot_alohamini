from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.motors import MotorNormMode
from lerobot.robots.alohamini.alohamini_host import build_observation_multipart, build_robot_metadata
from lerobot.robots.alohamini.kinematics import AlohaMiniJointMapping
from scripts.joycon_cartesian_control import (
    apply_local_rotation,
    arm_metadata,
    clip_arm_to_ik_limits,
    normalize_stick,
    observation_to_urdf,
    urdf_to_action,
    wait_for_robot_metadata,
)

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)


def test_stick_deadzone_and_rescaling():
    assert normalize_stick(2000, 0.25) == 0.0
    assert normalize_stick(2400, 0.25) == 0.0
    assert normalize_stick(3000, 0.25) == pytest.approx(1.0 / 3.0)
    assert normalize_stick(0, 0.25) == -1.0
    assert normalize_stick(4095, 0.25) == 1.0


def test_local_orientation_increment_stays_normalized_and_changes_orientation():
    result = apply_local_rotation([0.0, 0.0, 0.0, 1.0], 0.1, -0.2, 0.3)
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert result != [0.0, 0.0, 0.0, 1.0]


def test_arm_metadata_accepts_range_0_100_gripper():
    motors = {
        f"arm_right_{joint}": {
            "normalization": "range_0_100" if joint == "gripper" else "range_m100_100"
        }
        for joint in (*JOINTS, "gripper")
    }

    result = arm_metadata({"motors": motors}, "right")

    assert result["gripper"]["normalization"] == "range_0_100"


def test_out_of_range_measured_joint_is_clipped_for_dry_run_ik_seed():
    mapping = AlohaMiniJointMapping()
    values = np.asarray([mapping.calibration(joint).reference_q_rad for joint in JOINTS])
    wrist_index = JOINTS.index("wrist_flex")
    values[wrist_index] = mapping.calibration("wrist_flex").upper_rad + 0.1

    clipped, names = clip_arm_to_ik_limits(values, mapping)

    assert names == ["wrist_flex"]
    assert clipped[wrist_index] == mapping.calibration("wrist_flex").upper_rad


def test_metadata_wait_tolerates_first_fallback_observation():
    class FakeRobot:
        def __init__(self):
            self.robot_metadata = None
            self.calls = 0

        def get_observation(self):
            self.calls += 1
            if self.calls == 2:
                self.robot_metadata = {"schema_version": 1}
            return {"sample": self.calls}

    robot = FakeRobot()
    observation, metadata = wait_for_robot_metadata(robot, timeout_s=0.1)

    assert observation == {"sample": 2}
    assert metadata == {"schema_version": 1}


def test_state_only_observation_strips_camera_arrays_without_jpeg():
    parts = build_observation_multipart(
        {"arm_right_shoulder_pan.pos": 1.5, "forward": np.zeros((4, 4, 3), dtype=np.uint8)},
        camera_keys=("forward",),
        encoded_camera_keys=(),
    )

    assert len(parts) == 1
    state = json.loads(parts[0])
    assert state["arm_right_shoulder_pan.pos"] == 1.5
    assert state["_images"] == []
    assert "forward" not in state


def test_host_metadata_uses_live_bus_calibration():
    motor = SimpleNamespace(id=1, model="sts3250", norm_mode=MotorNormMode.RANGE_M100_100)
    calibration = SimpleNamespace(drive_mode=1, range_min=500, range_max=3500)
    bus = SimpleNamespace(motors={"arm_right_shoulder_pan": motor}, calibration={"arm_right_shoulder_pan": calibration})
    robot = SimpleNamespace(
        config=SimpleNamespace(robot_model="alohamini2pro"),
        left_bus=SimpleNamespace(motors={}, calibration={}),
        right_bus=bus,
    )

    metadata = build_robot_metadata(robot)

    assert metadata == {
        "schema_version": 1,
        "robot_model": "alohamini2pro",
        "motors": {
            "arm_right_shoulder_pan": {
                "id": 1,
                "model": "sts3250",
                "normalization": "range_m100_100",
                "drive_mode": 1,
                "range_min": 500,
                "range_max": 3500,
            }
        },
    }


def test_normalized_observation_urdf_action_round_trip():
    mapping = AlohaMiniJointMapping()
    metadata = {
        joint: {
            "normalization": "range_m100_100",
            "drive_mode": 0,
            "range_min": 0,
            "range_max": 4095,
        }
        for joint in JOINTS
    }
    q_expected = [mapping.calibration(joint).reference_q_rad for joint in JOINTS]
    action = urdf_to_action(q_expected, "right", metadata, mapping)
    observation = dict(action)

    q_recovered = observation_to_urdf(observation, "right", metadata, mapping)

    one_tick_rad = 2.0 * 3.141592653589793 / 4096
    assert q_recovered == pytest.approx(q_expected, abs=one_tick_rad)
