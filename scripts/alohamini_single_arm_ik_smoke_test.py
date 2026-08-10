#!/usr/bin/env python3
"""Minimal guarded Cartesian IK smoke test for one AlohaMini2Pro follower arm.

The default mode is read-only: it discovers only the configured IDs, reads the
current pose, solves IK and prints the proposed target.  Hardware writes require
the explicit ``--execute`` flag.  This is a commissioning tool, not the P3 ROS
hardware bridge.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence

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
CURRENT_MA_PER_RAW = 6.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--delta-mm",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 1.0),
        help="Cartesian offset in the selected arm base frame (default: 0 0 1).",
    )
    parser.add_argument("--execute", action="store_true", help="Enable torque and execute the guarded move.")
    parser.add_argument("--max-joint-delta-deg", type=float, default=1.0)
    parser.add_argument("--max-initial-limit-error-ticks", type=int, default=4)
    parser.add_argument("--max-tick-step", type=int, default=2)
    parser.add_argument("--period-s", type=float, default=0.05)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--current-limit-ma", type=float, default=1200.0)
    parser.add_argument("--final-tolerance-ticks", type=int, default=3)
    return parser.parse_args()


def shortest_tick_delta(target: int, current: int, period: int = 4096) -> int:
    return int((int(target) - int(current) + period // 2) % period - period // 2)


def build_tick_trajectory(
    current: dict[str, int], target: dict[str, int], max_tick_step: int, period: int = 4096
) -> list[dict[str, int]]:
    if set(current) != set(target):
        raise ValueError("Current and target motor sets differ")
    if max_tick_step < 1:
        raise ValueError("max_tick_step must be positive")
    deltas = {name: shortest_tick_delta(target[name], value, period) for name, value in current.items()}
    steps = max(1, math.ceil(max(abs(delta) for delta in deltas.values()) / max_tick_step))
    return [
        {name: int((value + round(deltas[name] * index / steps)) % period) for name, value in current.items()}
        for index in range(1, steps + 1)
    ]


def make_bus(port: str) -> FeetechMotorsBus:
    motors = {name: Motor(motor_id, model, None) for name, motor_id, model in ARM_SPECS}
    return FeetechMotorsBus(port, motors)


def read_registers(bus: FeetechMotorsBus, register: str) -> dict[str, int]:
    return {name: int(bus.read(register, name, normalize=False)) for name, *_ in ARM_SPECS}


def checked_seed(
    mapping: AlohaMiniJointMapping,
    raw_positions: dict[str, int],
    max_error_ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(
        [
            mapping.raw_tick_to_urdf(
                name,
                raw_positions[name],
                near_q_rad=mapping.calibration(name).reference_q_rad,
            )
            for name in mapping.joint_order
        ],
        dtype=float,
    )
    solver = AlohaMiniKinematics("right")
    seed = np.clip(observed, solver.lower_limits + 1e-4, solver.upper_limits - 1e-4)
    for index, name in enumerate(mapping.joint_order):
        corrected_tick = mapping.urdf_to_raw_tick(name, float(seed[index]))
        error_ticks = abs(shortest_tick_delta(corrected_tick, raw_positions[name]))
        if error_ticks > max_error_ticks:
            raise RuntimeError(
                f"{name} is outside the safe IK range by {error_ticks} ticks; reposition it manually"
            )
    return observed, seed


def solve_target(
    side: str,
    seed: np.ndarray,
    delta_mm: Sequence[float],
    max_joint_delta_deg: float,
) -> tuple[AlohaMiniKinematics, np.ndarray, np.ndarray]:
    if max_joint_delta_deg <= 0:
        raise ValueError("max_joint_delta_deg must be positive")
    kinematics = AlohaMiniKinematics(side)
    target_pose = kinematics.forward_kinematics(seed)
    target_pose[:3, 3] += np.asarray(delta_mm, dtype=float) / 1000.0
    result = kinematics.inverse_kinematics(
        target_pose,
        seed,
        max_iterations=300,
        max_joint_step_rad=math.radians(0.5),
    )
    if not result.success:
        raise RuntimeError(
            f"IK rejected target: {result.reason}, position_error={result.position_error_m * 1000:.3f} mm"
        )
    delta_deg = np.rad2deg(result.q_rad - seed)
    if float(np.max(np.abs(delta_deg))) > max_joint_delta_deg:
        raise RuntimeError(
            f"IK joint delta {float(np.max(np.abs(delta_deg))):.3f} deg exceeds {max_joint_delta_deg:.3f} deg"
        )
    return kinematics, target_pose, result.q_rad


def target_raw_positions(
    mapping: AlohaMiniJointMapping,
    current: dict[str, int],
    q_target: np.ndarray,
) -> dict[str, int]:
    target = dict(current)
    for name, value in zip(mapping.joint_order, q_target, strict=True):
        target[name] = mapping.urdf_to_raw_tick(name, float(value))
    return target


def ensure_current_is_safe(bus: FeetechMotorsBus, limit_ma: float) -> dict[str, float]:
    currents = read_registers(bus, "Present_Current")
    currents_ma = {name: abs(raw) * CURRENT_MA_PER_RAW for name, raw in currents.items()}
    over = {name: value for name, value in currents_ma.items() if value > limit_ma}
    if over:
        raise RuntimeError(f"Current limit exceeded: {over}")
    return currents_ma


def assert_torque_enabled(bus: FeetechMotorsBus) -> None:
    torque = read_registers(bus, "Torque_Enable")
    if any(value != 1 for value in torque.values()):
        raise RuntimeError(f"Motor torque did not remain enabled: {torque}")


def execute_trajectory(
    bus: FeetechMotorsBus,
    current: dict[str, int],
    target: dict[str, int],
    *,
    max_tick_step: int,
    period_s: float,
    settle_s: float,
    current_limit_ma: float,
    final_tolerance_ticks: int,
) -> dict[str, int]:
    if period_s <= 0 or settle_s < 0 or current_limit_ma <= 0:
        raise ValueError("period_s/current_limit_ma must be positive and settle_s non-negative")
    torque = read_registers(bus, "Torque_Enable")
    if any(value != 0 for value in torque.values()):
        raise RuntimeError(f"Expected every motor to start torque-disabled, got {torque}")
    modes = read_registers(bus, "Operating_Mode")
    if any(value != 0 for value in modes.values()):
        raise RuntimeError(f"Expected position mode (0) for every motor, got {modes}")

    torque_enabled = False
    peak_current_ma = dict.fromkeys(current, 0.0)

    def sample_safety() -> None:
        assert_torque_enabled(bus)
        for name, value in ensure_current_is_safe(bus, current_limit_ma).items():
            peak_current_ma[name] = max(peak_current_ma[name], value)

    try:
        bus.sync_write("Goal_Position", current, normalize=False)
        bus.enable_torque()
        torque_enabled = True
        sample_safety()
        for waypoint in build_tick_trajectory(current, target, max_tick_step):
            bus.sync_write("Goal_Position", waypoint, normalize=False)
            time.sleep(period_s)
            sample_safety()
        settle_deadline = time.monotonic() + settle_s
        while time.monotonic() < settle_deadline:
            time.sleep(min(period_s, max(0.0, settle_deadline - time.monotonic())))
            sample_safety()
        final = read_registers(bus, "Present_Position")
        errors = {name: abs(shortest_tick_delta(target[name], final[name])) for name in target}
        if max(errors.values()) > final_tolerance_ticks:
            raise RuntimeError(f"Final position tolerance exceeded: errors={errors}, final={final}")
        return final
    finally:
        print("peak_current_ma:", peak_current_ma)
        if torque_enabled:
            bus.disable_torque(num_retry=5)


def main() -> None:
    args = parse_args()
    mapping = AlohaMiniJointMapping()
    bus = make_bus(args.port)
    try:
        bus.connect(handshake=True)
        current = read_registers(bus, "Present_Position")
        observed, seed = checked_seed(mapping, current, args.max_initial_limit_error_ticks)
        kinematics, target_pose, q_target = solve_target(
            args.side,
            seed,
            args.delta_mm,
            args.max_joint_delta_deg,
        )
        target = target_raw_positions(mapping, current, q_target)
        print("current_ticks:", current)
        print("observed_q_rad:", observed.tolist())
        print("seed_correction_deg:", np.rad2deg(seed - observed).tolist())
        print("target_xyz_m:", target_pose[:3, 3].tolist())
        print("target_q_rad:", q_target.tolist())
        print("target_delta_deg:", np.rad2deg(q_target - observed).tolist())
        print("target_ticks:", target)
        print("trajectory_steps:", len(build_tick_trajectory(current, target, args.max_tick_step)))
        if not args.execute:
            print("READ-ONLY: pass --execute to perform this guarded move")
            return
        final = execute_trajectory(
            bus,
            current,
            target,
            max_tick_step=args.max_tick_step,
            period_s=args.period_s,
            settle_s=args.settle_s,
            current_limit_ma=args.current_limit_ma,
            final_tolerance_ticks=args.final_tolerance_ticks,
        )
        final_q = np.asarray(
            [
                mapping.raw_tick_to_urdf(
                    name,
                    final[name],
                    near_q_rad=float(q_target[index]),
                )
                for index, name in enumerate(mapping.joint_order)
            ]
        )
        final_tcp = kinematics.forward_kinematics(final_q)
        print("EXECUTED: torque restored to disabled")
        print("final_ticks:", final)
        print("final_tcp_xyz_m:", final_tcp[:3, 3].tolist())
        print("target_error_mm:", float(np.linalg.norm(final_tcp[:3, 3] - target_pose[:3, 3]) * 1000))
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
