#!/usr/bin/env python

import json
from types import SimpleNamespace

import pytest

from lerobot.robots.alohamini.alohamini import AlohaMini
from lerobot.robots.alohamini.alohamini_client import AlohaMiniClient
from lerobot.robots.alohamini.alohamini_host import (
    JointTrajectoryExecutor,
    build_observation_multipart,
    build_robot_metadata,
)


def make_executor() -> JointTrajectoryExecutor:
    return JointTrajectoryExecutor(
        max_velocity=10.0,
        max_acceleration=20.0,
        tracking_error_soft=2.0,
        tracking_error_hard=5.0,
    )


def test_trajectory_starts_at_feedback_and_limits_acceleration() -> None:
    executor = make_executor()
    executor.set_target({"arm_left_shoulder_pan.pos": 20.0})

    first = executor.step({"arm_left_shoulder_pan.pos": 3.0}, 0.1)
    second = executor.step({"arm_left_shoulder_pan.pos": first["arm_left_shoulder_pan.pos"]}, 0.1)

    assert first["arm_left_shoulder_pan.pos"] == pytest.approx(3.2)
    assert second["arm_left_shoulder_pan.pos"] == pytest.approx(3.6)


def test_tracking_gate_freezes_only_lagging_arm_group() -> None:
    executor = make_executor()
    executor.set_target(
        {
            "arm_left_shoulder_pan.pos": 20.0,
            "arm_left_elbow_flex.pos": 20.0,
            "arm_right_shoulder_pan.pos": 20.0,
            "arm_left_gripper.pos": 20.0,
        }
    )
    initial = {
        "arm_left_shoulder_pan.pos": 0.0,
        "arm_left_elbow_flex.pos": 0.0,
        "arm_right_shoulder_pan.pos": 0.0,
        "arm_left_gripper.pos": 0.0,
    }
    executor.step(initial, 0.1)

    measured = {
        **initial,
        # Both left body joints share a gate; only one needs to lag.
        "arm_left_shoulder_pan.pos": -5.0,
        "arm_left_elbow_flex.pos": 0.2,
        "arm_right_shoulder_pan.pos": 0.2,
        "arm_left_gripper.pos": 0.2,
    }
    gated = executor.step(measured, 0.1)

    assert gated["arm_left_shoulder_pan.pos"] == pytest.approx(0.2)
    assert gated["arm_left_elbow_flex.pos"] == pytest.approx(0.2)
    assert gated["arm_right_shoulder_pan.pos"] > 0.2
    assert gated["arm_left_gripper.pos"] > 0.2
    assert executor.last_progress_scale == 0.0


def test_hold_discards_pending_target_and_velocity() -> None:
    executor = make_executor()
    key = "arm_right_wrist_flex.pos"
    executor.set_target({key: 20.0})
    moving = executor.step({key: 0.0}, 0.1)

    executor.hold()
    held = executor.step({key: moving[key]}, 0.1)

    assert held == moving


def test_reversed_target_decelerates_without_command_jump() -> None:
    executor = make_executor()
    key = "arm_right_wrist_flex.pos"
    executor.set_target({key: 20.0})
    forward = executor.step({key: 0.0}, 0.1)

    executor.set_target({key: -20.0})
    reversing = executor.step({key: forward[key]}, 0.1)

    assert forward[key] == pytest.approx(0.2)
    assert reversing[key] == pytest.approx(forward[key])


def test_joint_diagnostics_separates_target_command_and_feedback() -> None:
    executor = make_executor()
    key = "arm_left_shoulder_pan.pos"
    executor.set_target({key: 20.0})
    command = executor.step({key: 3.0}, 0.1)[key]

    diagnostics = executor.joint_diagnostics(
        {key: 3.0}, {"arm_left_shoulder_pan": -130.0}
    )["arm_left_shoulder_pan"]

    assert diagnostics == pytest.approx(
        {
            "target": 20.0,
            "command": command,
            "measured": 3.0,
            "error": command - 3.0,
            "velocity": 2.0,
            "current_ma": -130.0,
            "progress_scale": 1.0,
        }
    )


def test_non_position_values_are_not_trajectory_targets() -> None:
    executor = make_executor()
    executor.set_target({"x.vel": 1.0, "lift_axis.height_mm": 100.0})

    assert executor.step({}, 0.1) == {}


def test_state_only_observation_has_no_jpeg_frames() -> None:
    parts = build_observation_multipart(
        {
            "arm_left_shoulder_pan.pos": 1.0,
            "forward": object(),
            "_robot_metadata": {"schema_version": 1},
        },
        camera_keys=("forward",),
        encoded_camera_keys=(),
    )

    assert len(parts) == 1
    state = json.loads(parts[0])
    assert state["_images"] == []
    assert state["_image_encoding"] == "jpeg"
    assert "forward" not in state
    assert state["_robot_metadata"] == {"schema_version": 1}


@pytest.mark.parametrize(
    ("include_cameras", "expected"),
    [(True, b"1:camera"), (False, b"1:state")],
)
def test_client_observation_token_selects_payload(
    include_cameras: bool, expected: bytes
) -> None:
    sent = []
    client = object.__new__(AlohaMiniClient)
    client._zmq = SimpleNamespace(NOBLOCK=1, ZMQError=RuntimeError)
    client._observation_request_id = 0
    client.zmq_observation_socket = SimpleNamespace(
        send=lambda token, flags: sent.append((token, flags))
    )

    token = AlohaMiniClient._send_observation_request(
        client, include_cameras=include_cameras
    )

    assert token == expected
    assert sent == [(expected, 1)]


def test_full_observation_keeps_camera_multipart_compatibility(monkeypatch) -> None:
    encoded = SimpleNamespace(tobytes=lambda: b"jpeg-data")
    monkeypatch.setattr(
        "lerobot.robots.alohamini.alohamini_host.cv2.imencode",
        lambda *_args, **_kwargs: (True, encoded),
    )

    timings = {}
    parts = build_observation_multipart(
        {"x.vel": 0.0, "forward": object()},
        camera_keys=("forward",),
        encoding_timings_ms=timings,
    )

    assert len(parts) == 3
    assert json.loads(parts[0])["_images"] == ["forward"]
    assert parts[1:] == [b"forward", b"jpeg-data"]
    assert timings["encode_forward"] >= 0.0


def test_control_observation_peeks_camera_cache_without_waiting_for_new_frame() -> None:
    class FakeCamera:
        def __init__(self) -> None:
            self.max_age_ms = None
            self.latest_timestamp = 12.5

        def read_latest(self, max_age_ms: int):
            self.max_age_ms = max_age_ms
            return "cached-frame"

    class FakeBus:
        def sync_read(self, register, motors):
            if register == "Present_Velocity":
                return dict.fromkeys(motors, 0.0)
            return {}

    robot = object.__new__(AlohaMini)
    robot.id = "test"
    robot.left_bus = FakeBus()
    robot.right_bus = None
    robot.left_arm_motors = []
    robot.right_arm_motors = []
    robot.base_motors = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]
    robot._wheel_raw_to_body = lambda *_args: {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }
    robot.lift = SimpleNamespace(contribute_observation=lambda _obs: None)
    robot.read_and_check_currents = lambda **_kwargs: {}
    camera = FakeCamera()
    robot.cameras = {"forward": camera}
    robot.logs = {}

    observation = AlohaMini.get_observation.__wrapped__(robot, include_cameras=True)

    assert observation["forward"] == "cached-frame"
    assert camera.max_age_ms == 500
    assert observation["_host_timing"]["camera_capture_monotonic_s"] == {
        "forward": 12.5
    }


def test_robot_metadata_describes_normalization_and_calibration() -> None:
    motor = SimpleNamespace(
        id=1,
        model="sts3250",
        norm_mode=SimpleNamespace(value="range_m100_100"),
    )
    calibration = SimpleNamespace(drive_mode=1, range_min=100, range_max=3900)
    left_bus = SimpleNamespace(
        motors={"arm_left_shoulder_pan": motor},
        calibration={"arm_left_shoulder_pan": calibration},
    )
    lift = SimpleNamespace(
        cfg=SimpleNamespace(soft_min_mm=0.0, soft_max_mm=600.0, descent_floor_mm=5.0)
    )
    robot = SimpleNamespace(
        left_bus=left_bus,
        right_bus=None,
        config=SimpleNamespace(robot_model="alohamini2pro"),
        lift=lift,
    )

    metadata = build_robot_metadata(robot)

    assert metadata["schema_version"] == 1
    assert metadata["robot_model"] == "alohamini2pro"
    assert metadata["motors"]["arm_left_shoulder_pan"] == {
        "id": 1,
        "model": "sts3250",
        "normalization": "range_m100_100",
        "drive_mode": 1,
        "range_min": 100,
        "range_max": 3900,
    }
    assert metadata["lift_axis"] == {
        "soft_min_mm": 0.0,
        "soft_max_mm": 600.0,
        "descent_floor_mm": 5.0,
    }


class FakeBus:
    def __init__(self) -> None:
        self.motors = {"arm_left_elbow_flex": object()}
        self.reads: list[str] = []

    def sync_read(self, register: str, motors: list[str]) -> dict[str, float]:
        self.reads.append(register)
        if register == "Present_Current":
            return dict.fromkeys(motors, 0.0)
        return dict.fromkeys(motors, 1.0)


def make_robot_feedback_stub(bus: FakeBus) -> AlohaMini:
    robot = object.__new__(AlohaMini)
    robot._feedback_currents_raw = {"arm_left_elbow_flex": 0.0}
    robot._feedback_positions = {"arm_left_elbow_flex": 1.0}
    robot._joint_current_limit_ma = 1800.0
    robot._joint_release_margin = 1.0
    robot._joint_hold_goal = {}
    robot._joint_hold_direction = {}
    return robot


def test_current_limiter_reuses_observe_act_feedback() -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)

    result = robot._limit_joint_goal_by_current(bus, {"arm_left_elbow_flex.pos": 2.0})

    assert result == {"arm_left_elbow_flex.pos": 2.0}
    assert bus.reads == []


def test_current_limiter_keeps_read_through_fallback() -> None:
    bus = FakeBus()
    robot = make_robot_feedback_stub(bus)
    robot._feedback_currents_raw.clear()
    robot._feedback_positions.clear()

    result = robot._limit_joint_goal_by_current(bus, {"arm_left_elbow_flex.pos": 2.0})

    assert result == {"arm_left_elbow_flex.pos": 2.0}
    assert bus.reads == ["Present_Current", "Present_Position"]


def make_gripper_feedback_stub(
    *, present: float, current_raw: float
) -> tuple[AlohaMini, FakeBus]:
    bus = FakeBus()
    bus.motors = {"arm_left_gripper": object()}
    robot = object.__new__(AlohaMini)
    robot._feedback_currents_raw = {"arm_left_gripper": current_raw}
    robot._feedback_positions = {"arm_left_gripper": present}
    robot._gripper_current_limit_ma = 500.0
    robot._gripper_release_margin = 1.0
    robot._gripper_hold_close_step = 3.0
    robot._gripper_open_direction = {"arm_left_gripper": 1.0}
    robot._gripper_hold_goal = {}
    robot._gripper_hold_direction = {}
    return robot, bus


def test_gripper_open_endpoint_overcurrent_releases_on_close_command() -> None:
    robot, bus = make_gripper_feedback_stub(present=90.0, current_raw=100.0)

    held = robot._limit_gripper_goal_by_current(
        bus, {"arm_left_gripper.pos": 100.0}
    )
    assert held["arm_left_gripper.pos"] == pytest.approx(90.0)

    robot._feedback_currents_raw["arm_left_gripper"] = 0.0
    closing = robot._limit_gripper_goal_by_current(
        bus, {"arm_left_gripper.pos": 0.0}
    )
    assert closing["arm_left_gripper.pos"] == pytest.approx(0.0)
    assert robot._gripper_hold_goal == {}


def test_gripper_closing_contact_retains_squeeze_and_releases_on_open() -> None:
    robot, bus = make_gripper_feedback_stub(present=30.0, current_raw=100.0)

    held = robot._limit_gripper_goal_by_current(
        bus, {"arm_left_gripper.pos": 0.0}
    )
    assert held["arm_left_gripper.pos"] == pytest.approx(27.0)

    robot._feedback_currents_raw["arm_left_gripper"] = 0.0
    opening = robot._limit_gripper_goal_by_current(
        bus, {"arm_left_gripper.pos": 100.0}
    )
    assert opening["arm_left_gripper.pos"] == pytest.approx(100.0)
    assert robot._gripper_hold_goal == {}
