from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("alohamini_description"))
    urdf = share / "alohamini2pro/alohamini2pro_kinematic.urdf"
    robot_description = urdf.read_text(encoding="utf-8")
    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
            Node(package="rviz2", executable="rviz2"),
        ]
    )
