"""ROS 2 FollowJointTrajectory bridge for one explicitly selected real arm."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from .mock_trajectory_bridge import ACCELERATION_LIMITS, ARM_SUFFIXES, VELOCITY_LIMITS, _seconds
from .trajectory_validation import TrajectoryPointData, interpolate_positions, validate_trajectory

MOTOR_SUFFIXES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")


class WorkerClient:
    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.lock = threading.Lock()
        self.counter = 0
        startup = self._read()
        if not startup.get("ok"):
            raise RuntimeError(startup.get("error", "worker startup failed"))
        self.state = startup["state"]

    def _read(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"hardware worker exited with code {self.process.poll()}")
        return json.loads(line)

    def request(self, command: str, **payload: object) -> dict:
        with self.lock:
            self.counter += 1
            request = {"id": self.counter, "command": command, **payload}
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            response = self._read()
            if response.get("id") != self.counter:
                raise RuntimeError(f"worker protocol mismatch: {response}")
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "worker request failed"))
            if "state" in response:
                self.state = response["state"]
            return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request("shutdown")
            except Exception:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class HardwareTrajectoryBridge(Node):
    def __init__(self) -> None:
        super().__init__("alohamini_hardware_trajectory_bridge")
        port = str(self.declare_parameter("port", "").value)
        side = str(self.declare_parameter("side", "right").value)
        execute = bool(self.declare_parameter("execute_hardware", False).value)
        worker_python = str(self.declare_parameter("worker_python", "python3").value)
        self.control_frequency = float(self.declare_parameter("control_frequency", 25.0).value)
        self.start_tolerance = float(self.declare_parameter("start_tolerance", 0.01).value)
        self.max_tick_step = int(self.declare_parameter("max_tick_step", 2).value)
        self.settle_s = float(self.declare_parameter("settle_s", 1.0).value)
        self.final_tolerance = float(self.declare_parameter("final_tolerance", 0.015).value)
        current_limit = float(self.declare_parameter("current_limit_ma", 1200.0).value)
        if not port:
            raise ValueError("port is required; no serial discovery is performed")
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if self.control_frequency <= 0 or self.max_tick_step < 1 or self.settle_s < 0:
            raise ValueError("control_frequency/max_tick_step must be positive and settle_s non-negative")
        if self.final_tolerance < 0:
            raise ValueError("final_tolerance must be non-negative")
        self.side = side
        self.execute_hardware = execute
        self.goal_lock = threading.Lock()
        self.goal_active = False
        self.names = tuple(f"{side}_{suffix}" for suffix in ARM_SUFFIXES)
        description = Path(get_package_share_directory("alohamini_description"))
        root = ET.parse(description / "alohamini2pro/alohamini2pro_moveit.urdf").getroot()
        self.position_limits = {}
        self.full_positions = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if joint.get("type") != "fixed" and name is not None:
                self.full_positions[name] = 0.0
            if name in self.names and limit is not None:
                self.position_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        if set(self.position_limits) != set(self.names):
            raise RuntimeError(f"missing URDF limits for {set(self.names) - set(self.position_limits)}")
        for arm_side in ("left", "right"):
            self.full_positions[f"{arm_side}_shoulder_lift"] = -1.571
            self.full_positions[f"{arm_side}_elbow_flex"] = 1.571
            self.full_positions[f"{arm_side}_gripper"] = 0.32
        worker = Path(get_package_share_directory("alohamini_control")) / "worker/hardware_worker.py"
        command = [worker_python, str(worker), "--port", port, "--side", side, "--current-limit-ma", str(current_limit)]
        if execute:
            command.append("--execute-hardware")
        self.worker = WorkerClient(command)
        self.callback_group = ReentrantCallbackGroup()
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / self.control_frequency, self._publish, callback_group=self.callback_group)
        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            f"/{side}_arm_controller/follow_joint_trajectory",
            goal_callback=self._goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            execute_callback=self._execute,
            callback_group=self.callback_group,
        )
        mode = "EXECUTION ENABLED" if execute else "READ-ONLY"
        self.get_logger().warning(f"Hardware bridge {mode}: side={side}, port={port}")
        self.get_logger().info(f"startup state: {self.worker.state}")

    def _points(self, request) -> tuple[TrajectoryPointData, ...]:
        return tuple(
            TrajectoryPointData(tuple(float(v) for v in point.positions), _seconds(point.time_from_start))
            for point in request.trajectory.points
        )

    def _current_positions(self) -> tuple[float, ...]:
        state = self.worker.request("state")["state"]
        return tuple(float(state["positions"][motor]) for motor in MOTOR_SUFFIXES)

    def _goal(self, request) -> GoalResponse:
        if not self.execute_hardware:
            self.get_logger().error("Rejected trajectory: bridge is read-only")
            return GoalResponse.REJECT
        with self.goal_lock:
            if self.goal_active:
                self.get_logger().error("Rejected trajectory: another hardware goal is active")
                return GoalResponse.REJECT
            self.goal_active = True
        try:
            points = self._points(request)
            current = self._current_positions()
        except Exception as exc:
            with self.goal_lock:
                self.goal_active = False
            self.get_logger().error(f"Rejected trajectory: failed to read hardware state: {exc}")
            return GoalResponse.REJECT
        effective_limits = dict(self.position_limits)
        for name, measured in zip(self.names, current, strict=True):
            lower, upper = effective_limits[name]
            # A measured startup just outside the calibrated range may hold or
            # move inward, but a trajectory may never move it farther outward.
            effective_limits[name] = (min(lower, measured), max(upper, measured))
        reason = validate_trajectory(
            request.trajectory.joint_names,
            points,
            expected_joint_names=self.names,
            position_limits=effective_limits,
            velocity_limits={name: VELOCITY_LIMITS[name.removeprefix(f"{self.side}_")] for name in self.names},
            acceleration_limits={name: ACCELERATION_LIMITS[name.removeprefix(f"{self.side}_")] for name in self.names},
        )
        if reason is None:
            error = max(
                abs(a - b)
                for a, b in zip(points[0].positions, current, strict=True)
            )
            if error > self.start_tolerance:
                reason = f"trajectory start differs from measured state by {error:.6f} rad"
        if reason:
            with self.goal_lock:
                self.goal_active = False
            self.get_logger().error(f"Rejected trajectory: {reason}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        points = self._points(goal_handle.request)
        period = 1.0 / self.control_frequency
        try:
            self.worker.request("arm")
            started = time.monotonic()
            while True:
                if goal_handle.is_cancel_requested:
                    self.worker.request("disarm")
                    goal_handle.canceled()
                    return FollowJointTrajectory.Result(error_code=0, error_string="canceled; torque disabled")
                elapsed = time.monotonic() - started
                desired = interpolate_positions(points, elapsed)
                motor_positions = dict(zip(MOTOR_SUFFIXES, desired, strict=True))
                state = self.worker.request("write", positions=motor_positions, max_tick_step=self.max_tick_step)["state"]
                actual = [float(state["positions"][name]) for name in MOTOR_SUFFIXES]
                feedback = FollowJointTrajectory.Feedback()
                feedback.joint_names = list(self.names)
                feedback.desired = JointTrajectoryPoint(positions=list(desired))
                feedback.actual = JointTrajectoryPoint(positions=actual)
                feedback.error = JointTrajectoryPoint(
                    positions=[d - a for d, a in zip(desired, actual, strict=True)]
                )
                goal_handle.publish_feedback(feedback)
                if elapsed >= points[-1].time_s:
                    break
                time.sleep(period)
            settle_deadline = time.monotonic() + self.settle_s
            while time.monotonic() < settle_deadline:
                time.sleep(min(period, max(0.0, settle_deadline - time.monotonic())))
                state = self.worker.request(
                    "write",
                    positions=dict(zip(MOTOR_SUFFIXES, desired, strict=True)),
                    max_tick_step=self.max_tick_step,
                )["state"]
            actual = [float(state["positions"][name]) for name in MOTOR_SUFFIXES]
            final_error = max(abs(d - a) for d, a in zip(desired, actual, strict=True))
            if final_error > self.final_tolerance:
                raise RuntimeError(
                    f"final tracking error {final_error:.6f} rad exceeds {self.final_tolerance:.6f}; "
                    f"ticks={state['ticks']}, goal_ticks={state['goal_ticks']}"
                )
            self.worker.request("disarm")
            goal_handle.succeed()
            return FollowJointTrajectory.Result(
                error_code=0,
                error_string=(
                    f"completed; final_error={final_error:.6f} rad, "
                    f"final_ticks={state['ticks']}; torque disabled"
                ),
            )
        except Exception as exc:
            with suppress(Exception):
                self.worker.request("disarm")
            goal_handle.abort()
            return FollowJointTrajectory.Result(error_code=-4, error_string=str(exc))
        finally:
            with self.goal_lock:
                self.goal_active = False

    def _publish(self) -> None:
        try:
            state = self.worker.request("state")["state"]
        except Exception as exc:
            self.get_logger().error(f"joint-state read failed: {exc}")
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        self.full_positions.update(
            {
                ros_name: float(state["positions"][motor])
                for ros_name, motor in zip(self.names, MOTOR_SUFFIXES, strict=True)
            }
        )
        message.name = list(self.full_positions)
        message.position = [self.full_positions[name] for name in message.name]
        self.publisher.publish(message)

    def destroy_node(self) -> None:
        self.server.destroy()
        self.worker.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareTrajectoryBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
