"""Dry-run PoseStamped to candidate JointTrajectory conversion.

This node never opens a serial port and never calls ``Robot.send_action``.
It is deliberately separated from the future hardware bridge so P0-P2 cannot
move the real robot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.robots.alohamini.kinematics import AlohaMiniKinematics


class AlohaMiniIKDryRunNode(Node):
    def __init__(self) -> None:
        super().__init__("alohamini_ik_dry_run")
        description_share = Path(get_package_share_directory("alohamini_description"))
        asset_dir = description_share / "alohamini2pro"
        self.kinematics = {side: AlohaMiniKinematics(side, asset_dir=asset_dir) for side in ("left", "right")}
        self.joint_positions: dict[str, float] = {}
        self.position_tolerance_m = float(self.declare_parameter("position_tolerance_m", 1e-4).value)
        self.orientation_tolerance_rad = float(
            self.declare_parameter("orientation_tolerance_rad", 1e-3).value
        )
        self.orientation_weight = float(self.declare_parameter("orientation_weight", 0.2).value)
        self.trajectory_duration_s = float(self.declare_parameter("trajectory_duration_s", 0.5).value)

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.trajectory_publishers = {}
        for side in ("left", "right"):
            self.create_subscription(
                PoseStamped,
                f"/alohamini/{side}/target_pose",
                lambda message, selected_side=side: self._on_target_pose(selected_side, message),
                10,
            )
            self.trajectory_publishers[side] = self.create_publisher(
                JointTrajectory,
                f"/alohamini/{side}/candidate_joint_trajectory",
                10,
            )
        self.diagnostics_publisher = self.create_publisher(DiagnosticArray, "/alohamini/ik_diagnostics", 10)
        self.get_logger().warning("IK node is DRY-RUN only; candidate trajectories are not executed")

    def _on_joint_state(self, message: JointState) -> None:
        if len(message.name) != len(message.position):
            self.get_logger().error("Rejected JointState with mismatched names and positions")
            return
        self.joint_positions.update(
            {name: float(position) for name, position in zip(message.name, message.position, strict=True)}
        )

    def _seed_for_side(self, side: str) -> np.ndarray | None:
        solver = self.kinematics[side]
        names = [solver.mapping.urdf_joint_name(side, joint) for joint in solver.joint_order]
        if any(name not in self.joint_positions for name in names):
            return None
        return np.asarray([self.joint_positions[name] for name in names], dtype=float)

    @staticmethod
    def _pose_matrix(message: PoseStamped) -> np.ndarray:
        pose = message.pose
        quaternion = np.asarray(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            dtype=float,
        )
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("Target quaternion is invalid")
        x, y, z, w = quaternion / norm
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return transform

    def _on_target_pose(self, side: str, message: PoseStamped) -> None:
        expected_frame = f"{side}_Base"
        if message.header.frame_id != expected_frame:
            self._publish_diagnostic(side, False, f"expected frame_id={expected_frame}")
            return
        seed = self._seed_for_side(side)
        if seed is None:
            self._publish_diagnostic(side, False, "complete measured JointState is not available")
            return
        try:
            target = self._pose_matrix(message)
            result = self.kinematics[side].inverse_kinematics(
                target,
                seed,
                position_tolerance_m=self.position_tolerance_m,
                orientation_tolerance_rad=self.orientation_tolerance_rad,
                orientation_weight=self.orientation_weight,
            )
        except Exception as error:
            self._publish_diagnostic(side, False, f"exception: {error}")
            return

        self._publish_diagnostic(
            side,
            result.success,
            result.reason,
            position_error_m=result.position_error_m,
            orientation_error_rad=result.orientation_error_rad,
            iterations=result.iterations,
            elapsed_s=result.elapsed_s,
        )
        if not result.success:
            return

        trajectory = JointTrajectory()
        trajectory.header = message.header
        trajectory.joint_names = [
            self.kinematics[side].mapping.urdf_joint_name(side, joint)
            for joint in self.kinematics[side].joint_order
        ]
        point = JointTrajectoryPoint()
        point.positions = result.q_rad.tolist()
        seconds = max(0.0, self.trajectory_duration_s)
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int((seconds - int(seconds)) * 1_000_000_000)
        trajectory.points = [point]
        self.trajectory_publishers[side].publish(trajectory)

    def _publish_diagnostic(self, side: str, success: bool, reason: str, **values: object) -> None:
        status = DiagnosticStatus()
        status.name = f"alohamini/{side}/inverse_kinematics"
        status.hardware_id = "dry_run"
        status.level = DiagnosticStatus.OK if success else DiagnosticStatus.ERROR
        status.message = reason
        status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostics_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AlohaMiniIKDryRunNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
