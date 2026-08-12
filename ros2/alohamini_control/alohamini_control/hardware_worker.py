"""LeRobot Python worker that exclusively owns one AlohaMini arm serial bus.

This module is intentionally launched as a subprocess by the ROS 2 bridge:
LeRobot uses Python 3.12 while ROS 2 Humble's rclpy extension uses Python 3.10.
The protocol is newline-delimited JSON on stdin/stdout.  Nothing is written to
the motors unless ``--execute-hardware`` was supplied and an explicit ``arm``
request passes every startup gate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

from lerobot.motors import Motor
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.alohamini.kinematics import AlohaMiniJointMapping

ARM_SPECS = (
    ("shoulder_pan", 1, "sts3250"),
    ("shoulder_lift", 2, "sts3095"),
    ("elbow_flex", 3, "sts3095"),
    ("wrist_flex", 4, "sts3250"),
    ("wrist_yaw", 5, "sts3250"),
    ("wrist_roll", 6, "sts3250"),
    ("gripper", 7, "sts3250"),
)
ARM_NAMES = tuple(name for name, *_ in ARM_SPECS[:-1])
CURRENT_MA_PER_RAW = 6.5


def _reply(request_id: object, *, ok: bool, **payload: object) -> None:
    print(json.dumps({"id": request_id, "ok": ok, **payload}, separators=(",", ":")), flush=True)


class HardwareWorker:
    def __init__(self, port: str, side: str, execute_hardware: bool, current_limit_ma: float) -> None:
        self.side = side
        self.execute_hardware = execute_hardware
        self.current_limit_ma = current_limit_ma
        self.mapping = AlohaMiniJointMapping()
        self.last_q: dict[str, float] = {}
        self.armed = False
        self.enabled_by_worker = False
        self.last_command: dict[str, int] | None = None
        self.arm_reference_q: dict[str, float] | None = None

        lock_path = Path("/tmp") / f"alohamini_{Path(os.path.realpath(port)).name}.lock"
        self.lock_file = lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"serial device is already owned (lock {lock_path})") from exc

        motors = {name: Motor(motor_id, model, None) for name, motor_id, model in ARM_SPECS}
        self.bus = FeetechMotorsBus(port, motors)
        self.bus.connect(handshake=True)

    def _read(self, register: str, names: tuple[str, ...] = ARM_NAMES) -> dict[str, int]:
        return {
            name: int(value)
            for name, value in self.bus.sync_read(register, list(names), normalize=False).items()
        }

    def state(self) -> dict[str, object]:
        ticks = self._read("Present_Position", tuple(name for name, *_ in ARM_SPECS))
        positions: dict[str, float] = {}
        for name in ARM_NAMES:
            near = self.last_q.get(name, self.mapping.calibration(name).reference_q_rad)
            positions[name] = self.mapping.raw_tick_to_urdf(name, ticks[name], near_q_rad=near)
        self.last_q = positions
        return {
            "positions": positions,
            "ticks": ticks,
            "goal_ticks": self._read("Goal_Position", tuple(name for name, *_ in ARM_SPECS)),
            "torque": self._read("Torque_Enable", tuple(name for name, *_ in ARM_SPECS)),
            "mode": self._read("Operating_Mode", tuple(name for name, *_ in ARM_SPECS)),
            "current_ma": {
                name: abs(raw) * CURRENT_MA_PER_RAW
                for name, raw in self._read("Present_Current", tuple(name for name, *_ in ARM_SPECS)).items()
            },
        }

    def arm(self) -> dict[str, object]:
        if not self.execute_hardware:
            raise RuntimeError("worker is read-only; restart with --execute-hardware")
        state = self.state()
        if any(value != 0 for value in state["torque"].values()):
            raise RuntimeError(f"expected all seven motors torque-disabled at startup, got {state['torque']}")
        if any(value != 0 for value in state["mode"].values()):
            raise RuntimeError(f"expected position mode (0), got {state['mode']}")
        if max(state["current_ma"].values()) > self.current_limit_ma:
            raise RuntimeError(f"current gate failed: {state['current_ma']}")
        current = {name: state["ticks"][name] for name in ARM_NAMES}
        self.bus.sync_write("Goal_Position", current, normalize=False)
        self.bus.enable_torque(list(ARM_NAMES))
        self.last_command = current
        self.arm_reference_q = {name: float(state["positions"][name]) for name in ARM_NAMES}
        self.enabled_by_worker = True
        self.armed = True
        return self.state()

    def write(self, q_target: dict[str, float], max_tick_step: int) -> dict[str, object]:
        if not self.armed:
            raise RuntimeError("hardware is not armed")
        if set(q_target) != set(ARM_NAMES):
            raise ValueError(f"expected joints {ARM_NAMES}, got {sorted(q_target)}")
        state = self.state()
        if any(state["torque"][name] != 1 for name in ARM_NAMES):
            raise RuntimeError(f"controlled motor torque dropped: {state['torque']}")
        if max(state["current_ma"][name] for name in ARM_NAMES) > self.current_limit_ma:
            raise RuntimeError(f"current gate failed: {state['current_ma']}")
        if self.arm_reference_q is None:
            raise RuntimeError("missing latched arm reference")
        target = {}
        for name in ARM_NAMES:
            q_rad = q_target[name]
            calibration = self.mapping.calibration(name)
            check_limits = True
            if calibration.lower_rad is not None and self.arm_reference_q[name] < calibration.lower_rad:
                if q_rad < self.arm_reference_q[name] - 1e-9:
                    raise RuntimeError(f"{name} command moves farther below its lower limit")
                check_limits = False
            if calibration.upper_rad is not None and self.arm_reference_q[name] > calibration.upper_rad:
                if q_rad > self.arm_reference_q[name] + 1e-9:
                    raise RuntimeError(f"{name} command moves farther above its upper limit")
                check_limits = False
            target[name] = self.mapping.urdf_to_raw_tick(name, q_rad, check_limits=check_limits)
        if self.last_command is None:
            raise RuntimeError("missing latched command state")
        deltas = {
            name: (target[name] - self.last_command[name] + 2048) % 4096 - 2048 for name in ARM_NAMES
        }
        if max(abs(value) for value in deltas.values()) > max_tick_step:
            raise RuntimeError(f"single-cycle tick step exceeds {max_tick_step}: {deltas}")
        self.bus.sync_write("Goal_Position", target, normalize=False)
        self.last_command = target
        return self.state()

    def disarm(self) -> dict[str, object]:
        if self.enabled_by_worker:
            self.bus.disable_torque(list(ARM_NAMES), num_retry=5)
        self.armed = False
        self.enabled_by_worker = False
        self.last_command = None
        self.arm_reference_q = None
        return self.state()

    def close(self) -> None:
        try:
            if self.enabled_by_worker:
                self.bus.disable_torque(list(ARM_NAMES), num_retry=5)
        finally:
            if self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)
            self.lock_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--execute-hardware", action="store_true")
    parser.add_argument("--current-limit-ma", type=float, default=1200.0)
    args = parser.parse_args()
    worker = HardwareWorker(args.port, args.side, args.execute_hardware, args.current_limit_ma)
    try:
        _reply("startup", ok=True, state=worker.state(), execute_hardware=args.execute_hardware)
        for line in sys.stdin:
            request: dict[str, object] = {}
            try:
                request = json.loads(line)
                command = request["command"]
                if command == "state":
                    result = worker.state()
                elif command == "arm":
                    result = worker.arm()
                elif command == "write":
                    result = worker.write(request["positions"], int(request["max_tick_step"]))
                elif command == "disarm":
                    result = worker.disarm()
                elif command == "shutdown":
                    _reply(request.get("id"), ok=True)
                    break
                else:
                    raise ValueError(f"unknown command {command!r}")
                _reply(request.get("id"), ok=True, state=result)
            except Exception as exc:
                _reply(request.get("id"), ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
