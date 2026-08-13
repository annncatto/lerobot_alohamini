#!/usr/bin/env python3
"""JSON-line MoveIt 2 FK/IK worker for the Joy-Con Cartesian controller.

Run this with ROS 2 Humble's Python 3.10.  Requests and responses use stdin and
stdout so the LeRobot Python 3.12 process does not need to import rclpy.
"""

from __future__ import annotations

import json
import sys
from collections import deque

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)

PREVIEW_HOME = {
    "root_x_axis_joint": 0.0,
    "root_y_axis_joint": 0.0,
    "root_z_rotation_joint": 0.0,
    "vertical_move": 0.0,
    "left_shoulder_pan": 0.0,
    "left_shoulder_lift": -1.571,
    "left_elbow_flex": 1.571,
    "left_wrist_flex": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_roll": 0.0,
    "left_gripper": 0.32,
    "right_shoulder_pan": 0.0,
    "right_shoulder_lift": -1.571,
    "right_elbow_flex": 1.571,
    "right_wrist_flex": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_roll": 0.0,
    "right_gripper": 0.32,
}


def joint_names(side: str) -> list[str]:
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    return [f"{side}_{suffix}" for suffix in ARM_SUFFIXES]


def fill_joint_state(state, side: str, positions: list[float]) -> None:
    arm_names = joint_names(side)
    if len(positions) != len(arm_names):
        raise ValueError(f"expected {len(arm_names)} seed positions")
    # Preview mode intentionally has no fixed /joint_states publisher before
    # this worker's first FK request. Supply every active joint so FK/IK does
    # not depend on stale planning-scene state to resolve right_Base/left_Base.
    complete = dict(PREVIEW_HOME)
    complete.update(zip(arm_names, (float(value) for value in positions), strict=True))
    state.joint_state.name = list(complete)
    state.joint_state.position = list(complete.values())


def pose_to_dict(pose) -> dict:
    return {
        "position": [pose.position.x, pose.position.y, pose.position.z],
        "orientation": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    }


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def rotate_vector(orientation: list[float], vector: list[float]) -> list[float]:
    x, y, z, w = orientation
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ]


def relative_pose(parent: dict, child: dict) -> dict:
    parent_inverse = [
        -parent["orientation"][0],
        -parent["orientation"][1],
        -parent["orientation"][2],
        parent["orientation"][3],
    ]
    offset = [child["position"][index] - parent["position"][index] for index in range(3)]
    return {
        "position": rotate_vector(parent_inverse, offset),
        "orientation": quaternion_multiply(parent_inverse, child["orientation"]),
    }


def compose_pose(parent: dict, child: dict) -> dict:
    rotated = rotate_vector(parent["orientation"], child["position"])
    return {
        "position": [parent["position"][index] + rotated[index] for index in range(3)],
        "orientation": quaternion_multiply(parent["orientation"], child["orientation"]),
    }


class MoveItWorker(Node):
    def __init__(self) -> None:
        super().__init__("alohamini_moveit_ik_worker")
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.fk = self.create_client(GetPositionFK, "/compute_fk")
        self.preview_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/alohamini/joycon_tcp_markers", 10
        )
        self.preview_positions = dict(PREVIEW_HOME)
        self.base_poses: dict[str, dict] = {}
        self.target_traces = {side: deque(maxlen=500) for side in ("left", "right")}
        self.candidate_traces = {side: deque(maxlen=500) for side in ("left", "right")}
        if not self.ik.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("MoveIt /compute_ik is unavailable; start plan_only.launch.py")
        if not self.fk.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("MoveIt /compute_fk is unavailable; start plan_only.launch.py")

    def call(self, client, request, timeout_s: float = 2.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("MoveIt service timed out")
        return response

    def fk_in_root(self, side: str, seed: list[float], links: list[str]) -> list[dict]:
        """Compute poses in the model frame without requiring an external TF tree."""
        request = GetPositionFK.Request()
        request.header.frame_id = "root"
        request.fk_link_names = links
        fill_joint_state(request.robot_state, side, seed)
        response = self.call(self.fk, request)
        if response.error_code.val != response.error_code.SUCCESS or len(response.pose_stamped) != len(links):
            raise RuntimeError(f"MoveIt FK failed with code {response.error_code.val}")
        return [pose_to_dict(item.pose) for item in response.pose_stamped]

    def base_pose(self, side: str, seed: list[float]) -> dict:
        if side not in self.base_poses:
            self.base_poses[side] = self.fk_in_root(side, seed, [f"{side}_Base"])[0]
        return self.base_poses[side]

    def compute_fk(self, request_data: dict) -> dict:
        side = request_data["side"]
        seed = request_data["seed"]
        base_pose, tcp_pose = self.fk_in_root(side, seed, [f"{side}_Base", f"{side}_tcp"])
        self.base_poses[side] = base_pose
        return {"pose": relative_pose(base_pose, tcp_pose)}

    def compute_ik(self, request_data: dict) -> dict:
        side = request_data["side"]
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = f"{side}_arm"
        ik.ik_link_name = f"{side}_tcp"
        seed = request_data["seed"]
        root_pose = compose_pose(self.base_pose(side, seed), request_data["pose"])
        ik.pose_stamped.header.frame_id = "root"
        position = root_pose["position"]
        orientation = root_pose["orientation"]
        ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = (
            float(value) for value in position
        )
        (
            ik.pose_stamped.pose.orientation.x,
            ik.pose_stamped.pose.orientation.y,
            ik.pose_stamped.pose.orientation.z,
            ik.pose_stamped.pose.orientation.w,
        ) = (float(value) for value in orientation)
        fill_joint_state(ik.robot_state, side, seed)
        ik.avoid_collisions = bool(request_data.get("avoid_collisions", False))
        timeout_s = float(request_data.get("timeout_s", 0.05))
        ik.timeout = Duration(sec=int(timeout_s), nanosec=int(timeout_s % 1.0 * 1_000_000_000))
        response = self.call(self.ik, request, timeout_s=max(2.0, timeout_s + 0.5))
        if response.error_code.val != response.error_code.SUCCESS:
            return {"success": False, "error_code": int(response.error_code.val)}
        solution = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
                strict=True,
            )
        )
        names = joint_names(side)
        if any(name not in solution for name in names):
            raise RuntimeError("MoveIt IK response omitted an arm joint")
        return {
            "success": True,
            "positions": [float(solution[name]) for name in names],
            "error_code": int(response.error_code.val),
        }

    def publish_preview(self, request_data: dict) -> dict:
        side = request_data["side"]
        positions = request_data["positions"]
        names = joint_names(side)
        if len(positions) != len(names):
            raise ValueError(f"expected {len(names)} preview arm positions")
        self.preview_positions.update(zip(names, (float(value) for value in positions), strict=True))
        self.preview_positions["root_x_axis_joint"] = float(request_data.get("base_x", 0.0))
        self.preview_positions["root_y_axis_joint"] = float(request_data.get("base_y", 0.0))
        self.preview_positions["root_z_rotation_joint"] = float(request_data.get("base_theta", 0.0))
        self.preview_positions["vertical_move"] = float(request_data.get("lift_m", 0.0))
        self.preview_positions[f"{side}_gripper"] = float(
            request_data.get("gripper_rad", self.preview_positions[f"{side}_gripper"])
        )

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self.preview_positions)
        message.position = list(self.preview_positions.values())
        self.preview_publisher.publish(message)
        self.publish_tcp_markers(
            side,
            request_data["measured_pose"],
            request_data["target_pose"],
            request_data["candidate_pose"],
            bool(request_data.get("reset_trace", False)),
        )
        return {"published": True}

    @staticmethod
    def _point(values: list[float]) -> Point:
        point = Point()
        point.x, point.y, point.z = (float(value) for value in values)
        return point

    @staticmethod
    def _rotate_vector(orientation: list[float], vector: tuple[float, float, float]) -> list[float]:
        x, y, z, w = (float(value) for value in orientation)
        vx, vy, vz = vector
        # Quaternion rotation expanded to avoid a numpy dependency in ROS Humble's Python.
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        return [
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ]

    def publish_tcp_markers(
        self,
        side: str,
        measured_pose: dict,
        target_pose: dict,
        candidate_pose: dict,
        reset_trace: bool,
    ) -> None:
        timestamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        for marker_id, (namespace, pose, color) in enumerate(
            (
                ("measured_tcp", measured_pose, (0.0, 0.7, 1.0, 0.35)),
                ("target_tcp", target_pose, (1.0, 0.8, 0.0, 1.0)),
                ("candidate_tcp", candidate_pose, (0.1, 1.0, 0.2, 1.0)),
            )
        ):
            marker = Marker()
            marker.header.frame_id = f"{side}_Base"
            marker.header.stamp = timestamp
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = self._point(pose["position"])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = (
                0.045
                if namespace == "measured_tcp"
                else 0.018
                if namespace == "target_tcp"
                else 0.024
            )
            marker.frame_locked = True
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            markers.markers.append(marker)

        origin = [float(value) for value in target_pose["position"]]
        axes = (
            ((1.0, 0.0, 0.0), (1.0, 0.1, 0.1, 1.0)),
            ((0.0, 1.0, 0.0), (0.1, 1.0, 0.1, 1.0)),
            ((0.0, 0.0, 1.0), (0.1, 0.3, 1.0, 1.0)),
        )
        for offset, (axis, color) in enumerate(axes, start=3):
            direction = self._rotate_vector(target_pose["orientation"], axis)
            endpoint = [origin[index] + 0.06 * direction[index] for index in range(3)]
            marker = Marker()
            marker.header.frame_id = f"{side}_Base"
            marker.header.stamp = timestamp
            marker.ns = "target_tcp_axes"
            marker.id = offset
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.points = [self._point(origin), self._point(endpoint)]
            marker.scale.x, marker.scale.y, marker.scale.z = 0.004, 0.008, 0.012
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker.frame_locked = True
            markers.markers.append(marker)

        label = Marker()
        label.header.frame_id = f"{side}_Base"
        label.header.stamp = timestamp
        label.ns = "target_tcp_label"
        label.id = 5
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = self._point([origin[0], origin[1], origin[2] + 0.04])
        label.pose.orientation.w = 1.0
        label.scale.z = 0.025
        label.color.r, label.color.g, label.color.b, label.color.a = (1.0, 0.8, 0.0, 1.0)
        label.text = "TCP target"
        label.frame_locked = True
        markers.markers.append(label)

        if reset_trace:
            self.target_traces[side].clear()
            self.candidate_traces[side].clear()
        self._append_trace_point(self.target_traces[side], target_pose["position"])
        self._append_trace_point(self.candidate_traces[side], candidate_pose["position"])
        markers.markers.extend(
            (
                self._trace_marker(
                    side, timestamp, "target_trace", 6, self.target_traces[side], (1.0, 0.8, 0.0, 0.9)
                ),
                self._trace_marker(
                    side,
                    timestamp,
                    "candidate_trace",
                    7,
                    self.candidate_traces[side],
                    (0.1, 1.0, 0.2, 0.9),
                ),
            )
        )

        self.marker_publisher.publish(markers)

    @staticmethod
    def _append_trace_point(trace: deque, position: list[float]) -> None:
        point = tuple(float(value) for value in position)
        if not trace or sum((point[index] - trace[-1][index]) ** 2 for index in range(3)) >= 1e-8:
            trace.append(point)

    def _trace_marker(self, side, timestamp, namespace, marker_id, trace, color) -> Marker:
        marker = Marker()
        marker.header.frame_id = f"{side}_Base"
        marker.header.stamp = timestamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.004
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.points = [self._point(point) for point in trace]
        marker.frame_locked = True
        return marker


def main() -> None:
    rclpy.init()
    node = MoveItWorker()
    print(json.dumps({"id": "startup", "ok": True}), flush=True)
    try:
        for line in sys.stdin:
            data = {}
            try:
                data = json.loads(line)
                if data["command"] == "fk":
                    result = node.compute_fk(data)
                elif data["command"] == "ik":
                    result = node.compute_ik(data)
                elif data["command"] == "preview":
                    result = node.publish_preview(data)
                elif data["command"] == "shutdown":
                    print(json.dumps({"id": data.get("id"), "ok": True}), flush=True)
                    break
                else:
                    raise ValueError(f"unknown command {data['command']!r}")
                print(json.dumps({"id": data.get("id"), "ok": True, **result}), flush=True)
            except Exception as exc:
                print(
                    json.dumps({"id": data.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                    flush=True,
                )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
