from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROS_ROOT = REPO_ROOT / "ros2"


def test_ros_packages_have_valid_names_and_build_types():
    expected = {
        "alohamini_description": "ament_cmake",
        "alohamini_kinematics": "ament_python",
        "alohamini_moveit_config": "ament_cmake",
        "alohamini_control": "ament_python",
    }
    for package_name, build_type in expected.items():
        root = ET.parse(ROS_ROOT / package_name / "package.xml").getroot()
        assert root.findtext("name") == package_name
        assert root.findtext("export/build_type") == build_type


def test_dry_run_node_has_no_hardware_write_path():
    path = ROS_ROOT / "alohamini_kinematics/alohamini_kinematics/ik_dry_run_node.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "send_action" not in called_attributes | called_names
    assert "FeetechMotorsBus" not in source
    assert "candidate_joint_trajectory" in source
    assert 'expected_frame = f"{side}_Base"' in source


def test_description_package_installs_generated_read_only_assets():
    cmake = (ROS_ROOT / "alohamini_description/CMakeLists.txt").read_text(encoding="utf-8")
    assert "sync_alohamini_kinematics_assets.py" in cmake
    assert "src/lerobot/robots/alohamini/assets/alohamini2pro" in cmake
    assert 'PATTERN "__pycache__" EXCLUDE' in cmake
    assert "alohamini2pro_meshes" in cmake
    assert "collision_meshes" in cmake


def test_moveit_launch_is_plan_only_by_default():
    source = (ROS_ROOT / "alohamini_moveit_config/launch/plan_only.launch.py").read_text()
    assert '"allow_trajectory_execution": False' in source
    assert "mock_trajectory_bridge" not in source
    assert "FeetechMotorsBus" not in source
    assert '"joycon_preview"' in source
    assert 'UnlessCondition(LaunchConfiguration("joycon_preview"))' in source
    assert 'config/joycon_preview.rviz' in source


def test_mock_bridge_has_no_hardware_backend():
    source = (ROS_ROOT / "alohamini_control/alohamini_control/mock_trajectory_bridge.py").read_text()
    assert "MOCK ONLY" in source
    assert "FeetechMotorsBus" not in source
    assert "/dev/tty" not in source


def test_hardware_bridge_is_default_locked_and_uses_separate_python_worker():
    bridge = (ROS_ROOT / "alohamini_control/alohamini_control/hardware_trajectory_bridge.py").read_text()
    worker = (ROS_ROOT / "alohamini_control/alohamini_control/hardware_worker.py").read_text()
    launch = (ROS_ROOT / "alohamini_control/launch/hardware_execution.launch.py").read_text()

    assert 'declare_parameter("execute_hardware", False)' in bridge
    assert 'declare_parameter("port", "")' in bridge
    assert 'start_new_session=True' in bridge
    assert 'if not self.execute_hardware:' in worker
    assert 'raise RuntimeError("worker is read-only' in worker
    assert 'default_value="false"' in launch
    assert 'LaunchConfiguration("execute_hardware")' in launch


def test_hardware_worker_has_independent_write_and_shutdown_gates():
    source = (ROS_ROOT / "alohamini_control/alohamini_control/hardware_worker.py").read_text()
    tree = ast.parse(source)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"arm", "write", "disarm", "close"} <= methods.keys()
    assert "single-cycle tick step exceeds" in source
    assert "current gate failed" in source
    assert "expected all seven motors torque-disabled" in source
    assert "disable_torque(list(ARM_NAMES)" in source


def test_joycon_cartesian_controller_is_dry_run_by_default():
    controller = (REPO_ROOT / "scripts/joycon_cartesian_control.py").read_text()
    worker = (REPO_ROOT / "scripts/alohamini_moveit_ik_worker.py").read_text()
    controller_tree = ast.parse(controller)

    execute_arguments = [
        node
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(isinstance(arg, ast.Constant) and arg.value == "--execute" for arg in node.args)
    ]
    assert len(execute_arguments) == 1
    assert any(
        keyword.arg == "action" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "store_true"
        for keyword in execute_arguments[0].keywords
    )
    assert "robot.send_action" in controller
    assert "if args.execute:" in controller
    assert "FeetechMotorsBus" not in controller
    assert "GetPositionIK" in worker
    assert "GetPositionFK" in worker
    assert 'create_publisher(JointState, "/joint_states"' in worker
    assert 'MarkerArray, "/alohamini/joycon_tcp_markers"' in worker
    assert 'data["command"] == "preview"' in worker
    assert "complete = dict(PREVIEW_HOME)" in worker
    assert 'request.header.frame_id = "root"' in worker
    assert "relative_pose(base_pose, tcp_pose)" in worker
    assert 'ik.pose_stamped.header.frame_id = "root"' in worker
    assert 'request_data.get("gripper_rad"' in worker
    assert 'Marker.LINE_STRIP' in worker
    assert 'request_data["candidate_pose"]' in worker
    assert "FeetechMotorsBus" not in worker
