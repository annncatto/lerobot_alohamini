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
