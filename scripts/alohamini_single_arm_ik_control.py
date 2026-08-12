#!/usr/bin/env python3
"""Direct interactive Cartesian control for one AlohaMini2Pro follower arm.

This is a bring-up utility for exercising the existing FK/IK implementation.
Each accepted IK result is converted to raw encoder ticks and streamed to the
servos as interpolated absolute ``Goal_Position`` waypoints.

Coordinates are expressed in the selected arm base frame. Interactive XYZ
arguments use millimetres.
"""

from __future__ import annotations

import argparse
import math
import shlex
import time

import numpy as np

from lerobot.motors import Motor
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.alohamini.kinematics import AlohaMiniJointMapping, AlohaMiniKinematics


ARM_SPECS = (
    ("shoulder_pan", 1, "sts3250"),
    ("shoulder_lift", 2, "sts3095"),
    ("elbow_flex", 3, "sts3095"),
    ("wrist_flex", 4, "sts3250"),
    ("wrist_yaw", 5, "sts3250"),
    ("wrist_roll", 6, "sts3250"),
    ("gripper", 7, "sts3250"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--mode",
        choices=("position", "pose"),
        default="position",
        help="position: constrain XYZ only; pose: also preserve the current TCP orientation.",
    )
    parser.add_argument(
        "--max-tick-step",
        type=int,
        default=2,
        help="Maximum absolute encoder change per trajectory waypoint (default: 2).",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=0.02,
        help="Seconds between trajectory waypoints (default: 0.02).",
    )
    return parser.parse_args()


def make_bus(port: str) -> FeetechMotorsBus:
    motors = {name: Motor(motor_id, model, None) for name, motor_id, model in ARM_SPECS}
    return FeetechMotorsBus(port, motors)


def shortest_tick_delta(target: int, current: int, period: int = 4096) -> int:
    return int((int(target) - int(current) + period // 2) % period - period // 2)


def build_tick_trajectory(
    current: dict[str, int],
    target: dict[str, int],
    max_tick_step: int,
    period: int = 4096,
) -> list[dict[str, int]]:
    if set(current) != set(target):
        raise ValueError("Current and target motor sets differ")
    if max_tick_step < 1:
        raise ValueError("max_tick_step must be positive")
    deltas = {name: shortest_tick_delta(target[name], value, period) for name, value in current.items()}
    steps = max(1, math.ceil(max(abs(delta) for delta in deltas.values()) / max_tick_step))
    return [
        {
            name: int((value + round(deltas[name] * index / steps)) % period)
            for name, value in current.items()
        }
        for index in range(1, steps + 1)
    ]


class DirectIKController:
    def __init__(self, port: str, side: str, mode: str, max_tick_step: int, period_s: float) -> None:
        if max_tick_step < 1:
            raise ValueError("max_tick_step must be positive")
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self.bus = make_bus(port)
        self.mapping = AlohaMiniJointMapping()
        self.kinematics = AlohaMiniKinematics(side)
        self.mode = mode
        self.max_tick_step = max_tick_step
        self.period_s = period_s

    def connect(self) -> None:
        self.bus.connect(handshake=True)
        current = self.read_ticks()
        # Avoid a jump when torque is enabled: latch the measured absolute pose
        # as the initial goal, then leave torque enabled for interactive control.
        self.bus.sync_write("Goal_Position", current, normalize=False)
        self.bus.enable_torque()

    def disconnect(self) -> None:
        if self.bus.is_connected:
            self.bus.disable_torque(num_retry=5)
            self.bus.disconnect(disable_torque=False)

    def read_ticks(self) -> dict[str, int]:
        return {
            name: int(self.bus.read("Present_Position", name, normalize=False))
            for name, *_ in ARM_SPECS
        }

    def read_q(self, ticks: dict[str, int]) -> np.ndarray:
        return np.asarray(
            [
                self.mapping.raw_tick_to_urdf(
                    name,
                    ticks[name],
                    near_q_rad=self.mapping.calibration(name).reference_q_rad,
                )
                for name in self.mapping.joint_order
            ],
            dtype=float,
        )

    def show(self) -> None:
        ticks = self.read_ticks()
        q = self.read_q(ticks)
        pose = self.kinematics.forward_kinematics(q)
        print("ticks:", ticks)
        print("q_deg:", dict(zip(self.mapping.joint_order, np.rad2deg(q).round(3), strict=True)))
        print("tcp_xyz_mm:", (pose[:3, 3] * 1000.0).round(3).tolist())
        print("mode:", self.mode)
        print(
            f"trajectory: max_tick_step={self.max_tick_step}, period_s={self.period_s}, "
            f"nominal_joint_rate_deg_s={self.max_tick_step * 360.0 / 4096 / self.period_s:.3f}"
        )

    def solve_and_write(self, target_pose: np.ndarray, seed_q: np.ndarray) -> None:
        orientation_weight = 0.0 if self.mode == "position" else 0.2
        # A sub-0.1 mm numerical residual is useful in unit tests but smaller
        # than the meaningful resolution of this servo arm. XYZ-only direct
        # control accepts a 0.5 mm residual; full-pose mode keeps the strict
        # core tolerance so orientation/position conflicts remain visible.
        position_tolerance_m = 5e-4 if self.mode == "position" else 1e-4
        # FK/target construction uses the actual measured configuration.  The
        # bounded IK core, however, cannot start even one encoder tick outside
        # its limits, so project only its numerical seed onto the valid set.
        # The resulting absolute trajectory still starts at the measured tick.
        ik_seed = np.clip(
            seed_q,
            self.kinematics.lower_limits + 1e-4,
            self.kinematics.upper_limits - 1e-4,
        )
        seed_correction_deg = np.rad2deg(ik_seed - seed_q)
        if np.any(np.abs(seed_correction_deg) > 1e-6):
            print("ik_seed_correction_deg:", seed_correction_deg.round(3).tolist())
        result = self.kinematics.inverse_kinematics(
            target_pose,
            ik_seed,
            position_tolerance_m=position_tolerance_m,
            orientation_weight=orientation_weight,
            max_iterations=1000,
            max_joint_step_rad=0.2,
        )
        print(
            f"ik: success={result.success} reason={result.reason} "
            f"iterations={result.iterations} position_error_mm={result.position_error_m * 1000:.3f} "
            f"orientation_error_deg={np.rad2deg(result.orientation_error_rad):.3f}"
        )
        if not result.success:
            return

        current = self.read_ticks()
        target = dict(current)
        for name, q_rad in zip(self.mapping.joint_order, result.q_rad, strict=True):
            target[name] = self.mapping.urdf_to_raw_tick(name, float(q_rad))
        print("target_q_deg:", np.rad2deg(result.q_rad).round(3).tolist())
        print("target_ticks:", target)
        trajectory = build_tick_trajectory(current, target, self.max_tick_step)
        print(
            f"trajectory_steps={len(trajectory)} "
            f"nominal_duration_s={len(trajectory) * self.period_s:.3f}"
        )
        for waypoint in trajectory:
            self.bus.sync_write("Goal_Position", waypoint, normalize=False)
            time.sleep(self.period_s)

    def move(self, delta_mm: np.ndarray) -> None:
        ticks = self.read_ticks()
        seed_q = self.read_q(ticks)
        target = self.kinematics.forward_kinematics(seed_q)
        target[:3, 3] += delta_mm / 1000.0
        self.solve_and_write(target, seed_q)

    def goto(self, xyz_mm: np.ndarray) -> None:
        ticks = self.read_ticks()
        seed_q = self.read_q(ticks)
        target = self.kinematics.forward_kinematics(seed_q)
        target[:3, 3] = xyz_mm / 1000.0
        self.solve_and_write(target, seed_q)


def parse_xyz(words: list[str]) -> np.ndarray:
    if len(words) != 3:
        raise ValueError("expected exactly three numbers: X Y Z (millimetres)")
    return np.asarray([float(value) for value in words], dtype=float)


def print_help() -> None:
    print("Commands:")
    print("  show                 read joint positions and TCP XYZ")
    print("  move DX DY DZ        move relative to the latest measured TCP pose (mm)")
    print("  goto X Y Z           move to an absolute TCP position in the arm base frame (mm)")
    print("  mode position|pose   switch between XYZ-only and fixed-orientation IK")
    print("  quit                 disable torque and exit")


def main() -> None:
    args = parse_args()
    controller = DirectIKController(
        args.port,
        args.side,
        args.mode,
        args.max_tick_step,
        args.period_s,
    )
    try:
        controller.connect()
        print_help()
        controller.show()
        while True:
            try:
                words = shlex.split(input("ik> "))
            except EOFError:
                break
            if not words:
                continue
            command, values = words[0].lower(), words[1:]
            try:
                if command in ("quit", "exit", "q"):
                    break
                if command == "help":
                    print_help()
                elif command == "show":
                    controller.show()
                elif command == "move":
                    controller.move(parse_xyz(values))
                elif command == "goto":
                    controller.goto(parse_xyz(values))
                elif command == "mode" and values in (["position"], ["pose"]):
                    controller.mode = values[0]
                    print("mode:", controller.mode)
                else:
                    print("Unknown command. Type 'help'.")
            except (RuntimeError, ValueError) as exc:
                print(f"ERROR: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
