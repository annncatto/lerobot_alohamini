"""Mock-only FollowJointTrajectory bridge.

This node never imports a motor bus and never opens a serial port.  It exists to
exercise MoveIt execution contracts before a separately reviewed hardware
backend is added.
"""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
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

from .trajectory_validation import TrajectoryPointData, interpolate_positions, validate_trajectory

ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)
VELOCITY_LIMITS = {
    "shoulder_pan": 0.5,
    "shoulder_lift": 0.4,
    "elbow_flex": 0.4,
    "wrist_flex": 0.7,
    "wrist_yaw_joint": 0.7,
    "wrist_roll": 0.7,
}
ACCELERATION_LIMITS = {name: 2.0 * value for name, value in VELOCITY_LIMITS.items()}


def _seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


class MockTrajectoryBridge(Node):
    def __init__(self) -> None:
        super().__init__("alohamini_mock_trajectory_bridge")
        self.control_frequency = float(self.declare_parameter("control_frequency", 25.0).value)
        self.start_tolerance = float(self.declare_parameter("start_tolerance", 0.01).value)
        if self.control_frequency <= 0.0 or self.start_tolerance < 0.0:
            raise ValueError("control_frequency must be positive and start_tolerance non-negative")

        description = Path(get_package_share_directory("alohamini_description"))
        urdf = description / "alohamini2pro/alohamini2pro_moveit.urdf"
        root = ET.parse(urdf).getroot()
        self.position_limits: dict[str, tuple[float, float]] = {}
        self.positions: dict[str, float] = {}
        self.positions_lock = threading.Lock()
        for joint in root.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            name = joint.get("name")
            limit = joint.find("limit")
            if name is None or limit is None:
                continue
            self.position_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
            self.positions[name] = 0.0
        for side in ("left", "right"):
            self.positions[f"{side}_shoulder_lift"] = -1.571
            self.positions[f"{side}_elbow_flex"] = 1.571
            self.positions[f"{side}_gripper"] = 0.32

        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.callback_group = ReentrantCallbackGroup()
        self.create_timer(
            1.0 / self.control_frequency,
            self._publish_joint_state,
            callback_group=self.callback_group,
        )
        self.servers = []
        for side in ("left", "right"):
            self.servers.append(
                ActionServer(
                    self,
                    FollowJointTrajectory,
                    f"/{side}_arm_controller/follow_joint_trajectory",
                    execute_callback=self._make_execute_callback(side),
                    goal_callback=self._make_goal_callback(side),
                    cancel_callback=self._cancel,
                    callback_group=self.callback_group,
                )
            )
        self.get_logger().warning("MOCK ONLY: trajectory goals update simulated joint states; no serial port is opened")

    def _names(self, side: str) -> tuple[str, ...]:
        return tuple(f"{side}_{suffix}" for suffix in ARM_SUFFIXES)

    def _point_data(self, request) -> tuple[TrajectoryPointData, ...]:
        return tuple(
            TrajectoryPointData(tuple(float(value) for value in point.positions), _seconds(point.time_from_start))
            for point in request.trajectory.points
        )

    def _make_execute_callback(self, side: str):
        def execute(goal_handle):
            return self._execute(side, goal_handle)

        return execute

    def _make_goal_callback(self, side: str):
        def goal(request):
            return self._goal(side, request)

        return goal

    def _goal(self, side: str, request) -> GoalResponse:
        names = self._names(side)
        points = self._point_data(request)
        reason = validate_trajectory(
            request.trajectory.joint_names,
            points,
            expected_joint_names=names,
            position_limits=self.position_limits,
            velocity_limits={name: VELOCITY_LIMITS[name.removeprefix(f"{side}_")] for name in names},
            acceleration_limits={
                name: ACCELERATION_LIMITS[name.removeprefix(f"{side}_")] for name in names
            },
        )
        if reason is None and points:
            with self.positions_lock:
                error = max(
                    abs(value - self.positions[name])
                    for name, value in zip(names, points[0].positions, strict=True)
                )
            if error > self.start_tolerance:
                reason = f"trajectory start differs from measured state by {error:.6f} rad"
        if reason is not None:
            self.get_logger().error(f"Rejected {side} trajectory: {reason}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute(self, side: str, goal_handle):
        names = self._names(side)
        points = self._point_data(goal_handle.request)
        started = time.monotonic()
        period = 1.0 / self.control_frequency
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return FollowJointTrajectory.Result(
                    error_code=FollowJointTrajectory.Result.SUCCESSFUL,
                    error_string="mock trajectory canceled",
                )
            elapsed = time.monotonic() - started
            desired = interpolate_positions(points, elapsed)
            with self.positions_lock:
                self.positions.update(dict(zip(names, desired, strict=True)))
                actual = [self.positions[name] for name in names]
            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = list(names)
            feedback.desired = JointTrajectoryPoint(positions=list(desired))
            feedback.actual = JointTrajectoryPoint(positions=actual)
            feedback.error = JointTrajectoryPoint(positions=[0.0] * len(names))
            goal_handle.publish_feedback(feedback)
            if elapsed >= points[-1].time_s:
                break
            time.sleep(period)
        goal_handle.succeed()
        return FollowJointTrajectory.Result(
            error_code=FollowJointTrajectory.Result.SUCCESSFUL,
            error_string="mock trajectory completed",
        )

    def _publish_joint_state(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        with self.positions_lock:
            message.name = list(self.positions)
            message.position = [self.positions[name] for name in message.name]
        self.publisher.publish(message)

    def destroy_node(self) -> None:
        for server in self.servers:
            server.destroy()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MockTrajectoryBridge()
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
