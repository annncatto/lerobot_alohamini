#!/usr/bin/env python3
"""Synchronize the audited RoboTwin AlohaMini2Pro kinematics assets.

The generated URDFs are intentionally mesh-free.  They preserve the complete
joint tree for FK, IK, TF and robot_state_publisher without copying the 132 MB
CAD meshes into LeRobot.  RoboTwin remains the source of truth; source hashes
are embedded in ``kinematics.json``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "ROBOTWIN_ALOHAMINI_ASSET",
        Path.home() / "RoboTwin/assets/embodiments/alohamini2pro",
    )
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src/lerobot/robots/alohamini/assets/alohamini2pro"

SOURCE_FILES = (
    "urdf/alohamini2pro.urdf",
    "urdf/alohamini2pro.srdf",
    "config/kinematics.yaml",
    "config/right_arm_kinematics.yaml",
    "config/hardware_joint_map_left.yaml",
    "config/hardware_joint_map_right.yaml",
    "config/joint_limits.yaml",
)

BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed generated assets differ from RoboTwin; do not write.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def resolve_shared_calibration(config_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    right_path = config_dir / "hardware_joint_map_right.yaml"
    left_path = config_dir / "hardware_joint_map_left.yaml"
    right = load_yaml(right_path)
    left_alias = load_yaml(left_path)
    if left_alias.get("side") != "left":
        raise ValueError("Left calibration alias must declare side: left")
    if left_alias.get("inherits") != right_path.name:
        raise ValueError("Left calibration must inherit the single measured right mapping")
    if left_alias.get("calibration_source_side") != "right":
        raise ValueError("Left calibration alias must declare calibration_source_side: right")
    if right.get("side") != "right" or not isinstance(right.get("joints"), dict):
        raise ValueError("The shared measured calibration source is incomplete")
    return right, left_alias


def _joint_signature(joint: ET.Element) -> tuple[dict[str, str], dict[str, str]]:
    origin = joint.find("origin")
    axis = joint.find("axis")
    return (
        {} if origin is None else dict(origin.attrib),
        {} if axis is None else dict(axis.attrib),
    )


def assert_symmetric_arm_chains(root: ET.Element) -> None:
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    urdf_suffixes = (*BODY_JOINTS[:4], "wrist_yaw_joint", BODY_JOINTS[5], "gripper")
    for suffix in urdf_suffixes:
        left = joints[f"left_{suffix}"]
        right = joints[f"right_{suffix}"]
        if _joint_signature(left) != _joint_signature(right):
            raise ValueError(f"Left/right URDF joint geometry differs for {suffix}")


def _descendants(root: ET.Element, root_link: str) -> tuple[set[str], list[ET.Element]]:
    joints_by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None:
            joints_by_parent.setdefault(parent.get("link"), []).append(joint)

    links = {root_link}
    selected_joints: list[ET.Element] = []
    stack = [root_link]
    while stack:
        parent_name = stack.pop()
        for joint in joints_by_parent.get(parent_name, []):
            child = joint.find("child").get("link")
            links.add(child)
            selected_joints.append(joint)
            stack.append(child)
    return links, selected_joints


def _strip_geometry(link: ET.Element) -> None:
    for tag in ("visual", "collision", "inertial"):
        for child in list(link.findall(tag)):
            link.remove(child)


def _matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    if sy > 1e-9:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        pitch = math.atan2(-float(matrix[2, 0]), sy)
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
        pitch = math.atan2(-float(matrix[2, 0]), sy)
        yaw = 0.0
    return roll, pitch, yaw


def _apply_shared_limits(joint: ET.Element, calibration: dict[str, Any]) -> None:
    name = joint.get("name")
    if not name or not name.startswith(("left_", "right_")):
        return
    short = name.split("_", 1)[1]
    if short == "wrist_yaw_joint":
        short = "wrist_yaw"
    entry = calibration["joints"].get(short)
    if entry is None:
        return
    limit = joint.find("limit")
    if limit is None:
        return
    if short == "wrist_roll":
        lower, upper = -math.pi, math.pi
    elif short == "gripper":
        lower, upper = float(entry["urdf_open_rad"]), float(entry["urdf_closed_rad"])
    else:
        lower = entry.get("safe_q_min_rad")
        upper = entry.get("safe_q_max_rad")
        if lower is None or upper is None:
            raise ValueError(f"Shared calibration is missing safe limits for {short}")
    limit.set("lower", f"{float(lower):.12g}")
    limit.set("upper", f"{float(upper):.12g}")


def _add_tcp_frame(robot: ET.Element, side: str, tool_frames: dict[str, Any]) -> None:
    delta = np.asarray(tool_frames["delta_matrix"], dtype=float)
    tcp_tool = np.asarray(tool_frames["tcp_tool_m"], dtype=float)
    tcp_fixed = delta @ tcp_tool
    roll, pitch, yaw = _matrix_to_rpy(delta)
    link_name = f"{side}_tcp"
    ET.SubElement(robot, "link", {"name": link_name})
    joint = ET.SubElement(robot, "joint", {"name": f"{side}_tcp_joint", "type": "fixed"})
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": " ".join(f"{value:.12g}" for value in tcp_fixed),
            "rpy": f"{roll:.12g} {pitch:.12g} {yaw:.12g}",
        },
    )
    ET.SubElement(joint, "parent", {"link": f"{side}_Fixed_Jaw"})
    ET.SubElement(joint, "child", {"link": link_name})


def _serialize_urdf(robot: ET.Element) -> bytes:
    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="utf-8", xml_declaration=True) + b"\n"


def build_full_kinematic_urdf(
    source_root: ET.Element,
    calibration: dict[str, Any],
    tool_frames: dict[str, Any],
) -> bytes:
    robot = copy.deepcopy(source_root)
    robot.set("name", "alohamini2pro_kinematic")
    for link in robot.findall("link"):
        _strip_geometry(link)
    for joint in robot.findall("joint"):
        _apply_shared_limits(joint, calibration)
    for side in ("left", "right"):
        _add_tcp_frame(robot, side, tool_frames)
    return _serialize_urdf(robot)


def build_arm_kinematic_urdf(
    source_root: ET.Element,
    side: str,
    calibration: dict[str, Any],
    tool_frames: dict[str, Any],
) -> bytes:
    root_link = f"{side}_Base"
    link_names, selected_joints = _descendants(source_root, root_link)
    robot = ET.Element("robot", {"name": f"alohamini2pro_{side}_kinematic"})
    for source_link in source_root.findall("link"):
        if source_link.get("name") not in link_names:
            continue
        link = copy.deepcopy(source_link)
        _strip_geometry(link)
        robot.append(link)
    for source_joint in selected_joints:
        joint = copy.deepcopy(source_joint)
        _apply_shared_limits(joint, calibration)
        robot.append(joint)
    _add_tcp_frame(robot, side, tool_frames)
    return _serialize_urdf(robot)


def build_kinematics_json(source_root: Path) -> bytes:
    source_paths = {relative: source_root / relative for relative in SOURCE_FILES}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RoboTwin source assets: {missing}")

    full_urdf_root = ET.parse(source_paths["urdf/alohamini2pro.urdf"]).getroot()
    assert_symmetric_arm_chains(full_urdf_root)
    calibration, left_alias = resolve_shared_calibration(source_root / "config")
    arm_kinematics = load_yaml(source_paths["config/right_arm_kinematics.yaml"])
    tool_frames = load_yaml(source_paths["config/kinematics.yaml"])["tool_frames"]

    rows = []
    for row in arm_kinematics["standard_dh"]["rows"]:
        copied = dict(row)
        copied["joint"] = copied["joint"].removeprefix("right_").replace("wrist_yaw_joint", "wrist_yaw")
        rows.append(copied)

    data = {
        "schema_version": 1,
        "asset": "alohamini2pro",
        "source_files": {
            relative: {"sha256": sha256(path), "size": path.stat().st_size}
            for relative, path in source_paths.items()
        },
        "shared_calibration": {
            "source_side": calibration["side"],
            "left_alias": left_alias,
            "ticks_per_revolution": calibration["ticks_per_revolution"],
            "joints": calibration["joints"],
        },
        "joint_order": list(BODY_JOINTS),
        "standard_dh": {
            "convention": arm_kinematics["standard_dh"]["convention"],
            "rows": rows,
            "base_transform": arm_kinematics["standard_dh"]["base_transform"],
            "tool_transform": arm_kinematics["standard_dh"]["tool_transform"],
        },
        "tcp": {
            "rotation_fixed_tool": tool_frames["delta_matrix"],
            "translation_tool_m": tool_frames["tcp_tool_m"],
            "translation_fixed_m": (
                np.asarray(tool_frames["delta_matrix"], dtype=float)
                @ np.asarray(tool_frames["tcp_tool_m"], dtype=float)
            ).tolist(),
        },
        "frames": {
            side: {
                "base": f"{side}_Base",
                "fixed_jaw": f"{side}_Fixed_Jaw",
                "tcp": f"{side}_tcp",
            }
            for side in ("left", "right")
        },
    }
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()


def generated_files(source_root: Path) -> dict[str, bytes]:
    full_source = source_root / "urdf/alohamini2pro.urdf"
    source_xml = ET.parse(full_source).getroot()
    assert_symmetric_arm_chains(source_xml)
    calibration, _ = resolve_shared_calibration(source_root / "config")
    tool_frames = load_yaml(source_root / "config/kinematics.yaml")["tool_frames"]
    return {
        "kinematics.json": build_kinematics_json(source_root),
        "alohamini2pro_kinematic.urdf": build_full_kinematic_urdf(source_xml, calibration, tool_frames),
        **{
            f"alohamini2pro_{side}_kinematic.urdf": build_arm_kinematic_urdf(
                source_xml, side, calibration, tool_frames
            )
            for side in ("left", "right")
        },
        "alohamini2pro.srdf": (source_root / "urdf/alohamini2pro.srdf").read_bytes(),
    }


def synchronize(source_root: Path, output_root: Path, check: bool) -> None:
    files = generated_files(source_root.resolve())
    stale = []
    for name, content in files.items():
        target = output_root.resolve() / name
        if check:
            if not target.is_file() or target.read_bytes() != content:
                stale.append(str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f"[WRITE] {target}")
    if stale:
        raise RuntimeError(f"Generated AlohaMini kinematics assets are stale: {stale}")
    if check:
        print(f"[OK] {len(files)} generated assets match RoboTwin")


def main() -> None:
    args = parse_args()
    synchronize(args.source_root, args.output_root, args.check)


if __name__ == "__main__":
    main()
