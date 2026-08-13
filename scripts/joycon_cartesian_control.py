#!/usr/bin/env python3
"""Joy-Con Cartesian teleoperation through ROS 2 MoveIt IK.

The Joy-Con/LeRobot process remains Python 3.12.  A separate ROS Humble Python
3.10 worker calls MoveIt's FK/IK services through a JSON-line pipe.  The robot
Host remains the only process that owns the motor buses.

Default mode is dry-run: observations and IK run, but no robot action is sent.
Use ``--execute`` only after checking the displayed measured joints and TCP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.robots.alohamini.kinematics import AlohaMiniJointMapping

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip", required=True)
    parser.add_argument("--robot-model", default="alohamini2pro")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--execute", action="store_true", help="Send arm/base/lift/gripper commands.")
    parser.add_argument(
        "--allow-unverified-robot-model",
        action="store_true",
        help="Allow execution when Host robot_model is not alohamini2pro (unsafe until calibrated).",
    )
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--tcp-speed-mm-s", type=float, default=25.0)
    parser.add_argument("--orientation-speed-deg-s", type=float, default=30.0)
    parser.add_argument("--max-cartesian-step-mm", type=float, default=2.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=2.0)
    parser.add_argument("--deadzone", type=float, default=0.25)
    parser.add_argument("--avoid-collisions", action="store_true")
    parser.add_argument("--base-speed", type=float, default=0.1)
    parser.add_argument("--base-rotation-speed", type=float, default=30.0)
    parser.add_argument("--lift-step-mm", type=float, default=5.0)
    parser.add_argument("--lift-min-mm", type=float, default=None)
    parser.add_argument("--lift-max-mm", type=float, default=None)
    parser.add_argument("--gripper-debounce-s", type=float, default=0.3)
    parser.add_argument("--metadata-timeout-s", type=float, default=5.0)
    return parser.parse_args()


def normalize_stick(raw: float, deadzone: float) -> float:
    value = max(-1.0, min(1.0, (float(raw) - 2000.0) / 2000.0))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply XYZW quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def apply_local_rotation(orientation: list[float], roll: float, pitch: float, yaw: float) -> list[float]:
    """Apply a small roll/pitch/yaw increment in the TCP local frame."""
    result = np.asarray(orientation, dtype=float)
    for axis, angle in (((1.0, 0.0, 0.0), roll), ((0.0, 1.0, 0.0), pitch), ((0.0, 0.0, 1.0), yaw)):
        if angle == 0.0:
            continue
        half = angle / 2.0
        delta = np.asarray([*(component * math.sin(half) for component in axis), math.cos(half)])
        result = quaternion_multiply(result, delta)
    return (result / np.linalg.norm(result)).tolist()


class MoveItClient:
    def __init__(self) -> None:
        command = (
            "source /opt/ros/humble/setup.bash && "
            f"source {REPO_ROOT}/ros2/.colcon/install/setup.bash && "
            "export RCUTILS_LOGGING_USE_STDOUT=0 && "
            f"exec /usr/bin/python3 {REPO_ROOT}/scripts/alohamini_moveit_ik_worker.py"
        )
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        self.process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=environment,
        )
        self.counter = 0
        startup = self._read()
        if not startup.get("ok"):
            raise RuntimeError(startup.get("error", "MoveIt worker failed to start"))

    def _read(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"MoveIt worker exited with code {self.process.poll()}")
        return json.loads(line)

    def request(self, command: str, **payload: object) -> dict:
        self.counter += 1
        request = {"id": self.counter, "command": command, **payload}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._read()
        if response.get("id") != self.counter:
            raise RuntimeError(f"MoveIt worker protocol mismatch: {response}")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "MoveIt request failed"))
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request("shutdown")
                self.process.wait(timeout=5.0)
            except Exception:
                self.process.terminate()


def arm_metadata(metadata: dict, side: str) -> dict[str, dict]:
    motors = metadata.get("motors", {})
    result = {}
    for joint in (*ARM_JOINTS, "gripper"):
        name = f"arm_{side}_{joint}"
        entry = motors.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"Host metadata is missing {name}; update/restart alohamini_host.py")
        if entry.get("normalization") not in ("range_m100_100", "range_0_100", "degrees"):
            raise RuntimeError(f"Unsupported normalization for {name}: {entry.get('normalization')}")
        result[joint] = entry
    return result


def observation_to_urdf(
    observation: dict,
    side: str,
    metadata: dict[str, dict],
    mapping: AlohaMiniJointMapping,
) -> np.ndarray:
    values = []
    for joint in ARM_JOINTS:
        key = f"arm_{side}_{joint}.pos"
        if key not in observation:
            raise RuntimeError(f"Observation is missing {key}")
        entry = metadata[joint]
        values.append(
            mapping.lerobot_to_urdf(
                joint,
                float(observation[key]),
                range_min=int(entry["range_min"]),
                range_max=int(entry["range_max"]),
                drive_mode=int(entry["drive_mode"]),
                normalization=entry["normalization"],
            )
        )
    return np.asarray(values, dtype=float)


def urdf_to_action(
    q_rad: np.ndarray,
    side: str,
    metadata: dict[str, dict],
    mapping: AlohaMiniJointMapping,
) -> dict[str, float]:
    result = {}
    for joint, value in zip(ARM_JOINTS, q_rad, strict=True):
        entry = metadata[joint]
        result[f"arm_{side}_{joint}.pos"] = mapping.urdf_to_lerobot(
            joint,
            float(value),
            range_min=int(entry["range_min"]),
            range_max=int(entry["range_max"]),
            drive_mode=int(entry["drive_mode"]),
            normalization=entry["normalization"],
        )
    return result


def clip_arm_to_ik_limits(
    q_rad: np.ndarray, mapping: AlohaMiniJointMapping
) -> tuple[np.ndarray, list[str]]:
    """Return an IK-valid seed and the names of joints that were clipped."""
    clipped = np.asarray(q_rad, dtype=float).copy()
    clipped_names = []
    for index, joint in enumerate(ARM_JOINTS):
        calibration = mapping.calibration(joint)
        bounded = float(
            np.clip(clipped[index], calibration.lower_rad, calibration.upper_rad)
        )
        if not math.isclose(bounded, clipped[index], abs_tol=1e-9):
            clipped_names.append(joint)
            clipped[index] = bounded
    return clipped, clipped_names


def wait_for_robot_metadata(robot, timeout_s: float) -> tuple[dict, dict]:
    """Wait for a fresh Host frame carrying calibration metadata.

    ``get_observation`` may return its fallback state when the first request
    after connect times out. Treating that one transient frame as proof of an
    old Host made controller startup nondeterministic.
    """
    deadline = time.monotonic() + timeout_s
    latest_observation = {}
    while time.monotonic() < deadline:
        latest_observation = robot.get_observation()
        metadata = robot.robot_metadata
        if isinstance(metadata, dict):
            return latest_observation, metadata
    raise RuntimeError(
        f"Host did not provide motor metadata within {timeout_s:.1f}s; "
        "deploy/restart the updated alohamini_host.py"
    )


def buttons(joy) -> dict[str, bool]:
    return {
        "x": bool(joy.joycon.get_button_x()),
        "b": bool(joy.joycon.get_button_b()),
        "y": bool(joy.joycon.get_button_y()),
        "a": bool(joy.joycon.get_button_a()),
        "r": bool(joy.joycon.get_button_r()),
        "zr": bool(joy.joycon.get_button_zr()),
        "home": bool(joy.joycon.get_button_home()),
        "plus": bool(joy.joycon.get_button_plus()),
        "sl": bool(joy.joycon.get_button_right_sl()),
        "sr": bool(joy.joycon.get_button_right_sr()),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.deadzone < 1:
        raise ValueError("deadzone must be in [0, 1)")
    for name in (
        "control_hz",
        "tcp_speed_mm_s",
        "orientation_speed_deg_s",
        "max_cartesian_step_mm",
        "max_joint_step_deg",
        "metadata_timeout_s",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    try:
        import joyconrobotics
    except ImportError as exc:
        raise RuntimeError("Install the team's joyconrobotics package in the LeRobot environment") from exc

    config = AlohaMiniClientConfig(
        remote_ip=args.remote_ip,
        id="joycon_cartesian_controller",
        robot_model=args.robot_model,
        request_cameras=False,
        observation_request_window=1,
        cameras={},
    )
    robot = AlohaMiniClient(config=config)
    moveit = None
    joy = None
    try:
        robot.connect()
        observation, metadata_packet = wait_for_robot_metadata(robot, args.metadata_timeout_s)
        host_model = metadata_packet.get("robot_model")
        if host_model != args.robot_model:
            raise RuntimeError(f"Client robot_model={args.robot_model}, Host robot_model={host_model}")
        if args.execute and host_model != "alohamini2pro" and not args.allow_unverified_robot_model:
            raise RuntimeError(
                "MoveIt asset/mapping is calibrated for alohamini2pro; execution on "
                f"{host_model} requires --allow-unverified-robot-model after a separate zero/direction audit"
            )
        metadata = arm_metadata(metadata_packet, args.side)
        lift_metadata = metadata_packet.get("lift_axis", {})
        lift_min_mm = (
            float(args.lift_min_mm)
            if args.lift_min_mm is not None
            else float(lift_metadata.get("soft_min_mm", 0.0))
        )
        lift_max_mm = (
            float(args.lift_max_mm)
            if args.lift_max_mm is not None
            else float(lift_metadata.get("soft_max_mm", 600.0))
        )
        if lift_min_mm >= lift_max_mm:
            raise RuntimeError(f"invalid lift range [{lift_min_mm}, {lift_max_mm}] mm")
        mapping = AlohaMiniJointMapping()
        q_measured = observation_to_urdf(observation, args.side, metadata, mapping)

        moveit = MoveItClient()
        measured_pose = moveit.request("fk", side=args.side, seed=q_measured.tolist())["pose"]
        q_target, clipped_joints = clip_arm_to_ik_limits(q_measured, mapping)
        if clipped_joints:
            details = ", ".join(
                f"{joint}={q_measured[index]:.4f}->{q_target[index]:.4f}rad"
                for index, joint in enumerate(ARM_JOINTS)
                if joint in clipped_joints
            )
            if args.execute:
                raise RuntimeError(
                    "Measured arm state is outside the audited MoveIt limits; refuse execution: "
                    + details
                )
            print(f"WARNING: clipping dry-run IK seed to MoveIt limits: {details}")
        # Keep three poses distinct in the preview:
        # measured_pose: latest real robot state (cyan),
        # accepted_pose: latest IK-solvable command (green),
        # attempted_pose: the latest joystick request, including rejected ones (yellow).
        accepted_pose = moveit.request("fk", side=args.side, seed=q_target.tolist())["pose"]
        attempted_pose = accepted_pose
        joy = joyconrobotics.JoyconRobotics(device="right")
        target_gripper = float(observation.get(f"arm_{args.side}_gripper.pos", 0.0))
        target_height = float(
            np.clip(observation.get("lift_axis.height_mm", lift_min_mm), lift_min_mm, lift_max_mm)
        )
        last_toggle = 0.0
        last_mode_toggle = 0.0
        previous_plus = False
        previous_zr = False
        control_mode = "position"
        last_time = time.monotonic()
        previous_sequence = robot.observation_sequence
        preview_base = np.zeros(3)
        preview_counter = 0
        mode = "EXECUTE" if args.execute else "DRY-RUN"
        print(f"{mode}: TCP={np.round(accepted_pose['position'], 4).tolist()} m")
        print("Position mode: stick TCP X/Y; hold R + stick up/down: TCP Z; R alone: lift up")
        print("Plus toggles orientation mode: stick pitch/yaw; SL/SR roll")
        print("X/B base forward/back; A/Y left/right; R+A/Y rotate; ZR gripper; Home re-latches TCP")

        while True:
            loop_started = time.monotonic()
            observation = robot.get_observation()
            if robot.observation_sequence == previous_sequence:
                continue
            previous_sequence = robot.observation_sequence
            q_measured = observation_to_urdf(observation, args.side, metadata, mapping)
            now = time.monotonic()
            dt = min(0.1, max(0.0, now - last_time))
            last_time = now

            stick_v, stick_h, stick_pressed = joy.get_stick()
            stick_y = normalize_stick(stick_v, args.deadzone)
            stick_x = normalize_stick(stick_h, args.deadzone)
            button = buttons(joy)
            mode_toggled = False
            if button["plus"] and not previous_plus and now - last_mode_toggle > 0.5:
                control_mode = "orientation" if control_mode == "position" else "position"
                last_mode_toggle = now
                mode_toggled = True
            previous_plus = button["plus"]
            if button["home"]:
                q_target, _ = clip_arm_to_ik_limits(q_measured, mapping)
                accepted_pose = moveit.request("fk", side=args.side, seed=q_target.tolist())["pose"]
                attempted_pose = accepted_pose

            delta = np.zeros(3)
            rotation = np.zeros(3)
            max_step_m = args.max_cartesian_step_mm / 1000.0
            speed_m_s = args.tcp_speed_mm_s / 1000.0
            if control_mode == "orientation":
                if not mode_toggled and not button["home"]:
                    angular_speed = math.radians(args.orientation_speed_deg_s)
                    rotation[0] = (float(button["sr"]) - float(button["sl"])) * angular_speed * dt
                    rotation[1] = stick_y * angular_speed * dt
                    rotation[2] = -stick_x * angular_speed * dt
            elif button["r"] and (stick_x != 0.0 or stick_y != 0.0):
                delta[2] = stick_y * speed_m_s * dt
            elif not button["home"]:
                delta[0] = stick_y * speed_m_s * dt
                delta[1] = -stick_x * speed_m_s * dt
            norm = float(np.linalg.norm(delta))
            if norm > max_step_m:
                delta *= max_step_m / norm

            q_seed = q_measured if args.execute else q_target
            q_command = q_seed
            ik_status = "IDLE"
            target_changed = norm > 0.0 or bool(np.any(rotation))
            if target_changed and not button["home"]:
                attempted_pose = {
                    "position": (np.asarray(accepted_pose["position"]) + delta).tolist(),
                    "orientation": apply_local_rotation(accepted_pose["orientation"], *rotation),
                }
                ik = moveit.request(
                    "ik",
                    side=args.side,
                    seed=q_seed.tolist(),
                    pose=attempted_pose,
                    avoid_collisions=args.avoid_collisions,
                )
                if bool(ik["success"]):
                    q_candidate = np.asarray(ik["positions"], dtype=float)
                    joint_delta = q_candidate - q_seed
                    if float(np.max(np.abs(joint_delta))) <= math.radians(args.max_joint_step_deg):
                        q_command = q_candidate
                        q_target = q_candidate
                        accepted_pose = attempted_pose
                        ik_status = "OK"
                    else:
                        ik_status = "REJECT_STEP"
                else:
                    ik_status = f"REJECT_{ik.get('error_code', 'UNKNOWN')}"

            # LeRobot body convention: +x forward, +y left, +theta left turn.
            base_vx = args.base_speed if button["x"] else -args.base_speed if button["b"] else 0.0
            base_vy = (
                args.base_speed
                if button["a"] and not button["r"]
                else -args.base_speed
                if button["y"] and not button["r"]
                else 0.0
            )
            base_theta = (
                args.base_rotation_speed
                if button["r"] and button["a"]
                else -args.base_rotation_speed
                if button["r"] and button["y"]
                else 0.0
            )
            if (
                control_mode == "position"
                and button["r"]
                and stick_x == 0.0
                and stick_y == 0.0
                and not button["y"]
                and not button["a"]
            ):
                target_height = min(lift_max_mm, target_height + args.lift_step_mm)
            if stick_pressed:
                target_height = max(lift_min_mm, target_height - args.lift_step_mm)
            if button["zr"] and not previous_zr and now - last_toggle > args.gripper_debounce_s:
                target_gripper = 0.0 if target_gripper > 50.0 else 100.0
                last_toggle = now
            previous_zr = button["zr"]

            # The authoritative CAD/URDF uses +X left and -Y forward, while
            # LeRobot uses +x forward and +y left.
            preview_base += np.asarray([base_vy, -base_vx, math.radians(base_theta)]) * dt
            preview_counter += 1
            if preview_counter % 5 == 0:
                measured_pose = moveit.request("fk", side=args.side, seed=q_measured.tolist())["pose"]
            moveit.request(
                "preview",
                side=args.side,
                positions=q_command.tolist(),
                base_x=float(preview_base[0]),
                base_y=float(preview_base[1]),
                base_theta=float(preview_base[2]),
                lift_m=(target_height - (lift_min_mm + lift_max_mm) / 2.0) / 1000.0,
                gripper_rad=mapping.lerobot_to_urdf(
                    "gripper",
                    target_gripper,
                    range_min=int(metadata["gripper"]["range_min"]),
                    range_max=int(metadata["gripper"]["range_max"]),
                    drive_mode=int(metadata["gripper"]["drive_mode"]),
                    normalization=metadata["gripper"]["normalization"],
                ),
                target_pose=attempted_pose,
                measured_pose=measured_pose,
                candidate_pose=accepted_pose,
                reset_trace=button["home"],
            )

            if args.execute:
                action = urdf_to_action(q_command, args.side, metadata, mapping)
                action[f"arm_{args.side}_gripper.pos"] = target_gripper
                action.update(
                    {
                        "x.vel": base_vx,
                        "y.vel": base_vy,
                        "theta.vel": base_theta,
                        "lift_axis.height_mm": target_height,
                    }
                )
                robot.send_action(action)

            print(
                f"\r{mode} {control_mode.upper()} IK={ik_status} "
                f"TCP(mm)={np.round(np.asarray(accepted_pose['position']) * 1000.0, 1).tolist()} "
                f"stick=({stick_x:+.2f},{stick_y:+.2f}) "
                f"lift={target_height:.0f}[{lift_min_mm:.0f},{lift_max_mm:.0f}] obs={previous_sequence}",
                end="",
                flush=True,
            )
            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, 1.0 / args.control_hz - elapsed))
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        if moveit is not None:
            moveit.close()
        if robot.is_connected:
            if args.execute:
                robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
            robot.disconnect()


if __name__ == "__main__":
    main()
