"""Single-arm MoveIt execution with a default-read-only real serial bridge."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description() -> LaunchDescription:
    description = Path(get_package_share_directory("alohamini_description")) / "alohamini2pro"
    moveit_package = Path(get_package_share_directory("alohamini_moveit_config"))
    moveit_config = (
        MoveItConfigsBuilder("alohamini2pro", package_name="alohamini_moveit_config")
        .robot_description(file_path=str(description / "alohamini2pro_moveit.urdf"))
        .robot_description_semantic(file_path=str(description / "alohamini2pro.srdf"))
        .robot_description_kinematics()
        .joint_limits()
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .trajectory_execution(file_path="config/moveit_controllers.yaml", moveit_manage_controllers=False)
        .planning_scene_monitor(
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value=""),
            DeclareLaunchArgument("side", default_value="right"),
            DeclareLaunchArgument("execute_hardware", default_value="false"),
            DeclareLaunchArgument(
                "worker_python",
                default_value="/home/anncatto/miniconda3/envs/lerobot_alohamini/bin/python",
            ),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="alohamini_control",
                executable="hardware_trajectory_bridge",
                output="screen",
                parameters=[
                    {
                        "port": LaunchConfiguration("port"),
                        "side": LaunchConfiguration("side"),
                        "execute_hardware": LaunchConfiguration("execute_hardware"),
                        "worker_python": LaunchConfiguration("worker_python"),
                    }
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[moveit_config.robot_description],
                output="screen",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {
                        "allow_trajectory_execution": LaunchConfiguration("execute_hardware"),
                        "publish_robot_description_semantic": True,
                        "monitor_dynamics": False,
                    },
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="moveit_rviz",
                output="log",
                arguments=["-d", str(moveit_package / "config/moveit.rviz")],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.planning_pipelines,
                    moveit_config.joint_limits,
                ],
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
        ]
    )
