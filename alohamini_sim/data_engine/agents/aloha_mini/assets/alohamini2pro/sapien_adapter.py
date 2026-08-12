"""SAPIEN runtime adapter for the AlohaMini2Pro description asset.

The adapter owns articulation loading, initial SI-unit qpos, per-group drives,
gravity compensation and normalized gripper-command conversion. It deliberately
does not import RoboTwin tasks or cuRobo.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sapien
import yaml


class AlohaMini2ProSapienAdapter:
    def __init__(self, asset_dir: str | Path | None = None):
        self.asset_dir = Path(asset_dir or Path(__file__).resolve().parent).resolve()
        self.config_dir = self.asset_dir / "config"
        self.articulation_config = self._load_yaml("sapien_articulation.yaml")
        self.control_config = self._load_yaml("sapien_control.yaml")
        self.gripper_config = self._load_yaml("gripper.yaml")
        self.physical_config = self._load_yaml("physical_parameters.yaml")
        self.joint_limits_config = self._load_yaml("joint_limits.yaml")

        self.urdf_path = (self.config_dir / self.articulation_config["urdf_path"]).resolve()
        self.srdf_path = (self.config_dir / self.articulation_config["srdf_path"]).resolve()
        self.articulation = None
        self.joint_names: list[str] = []
        self.joints_by_name = {}
        self._temporary_urdf: Path | None = None
        self.validate_description_files()

    def _load_yaml(self, name: str) -> dict:
        path = self.config_dir / name
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid adapter configuration: {path}")
        return data

    @property
    def timestep(self) -> float:
        return float(self.articulation_config["simulation"]["timestep"])

    @property
    def gravity_compensation(self) -> bool:
        return bool(self.articulation_config["simulation"].get("gravity_compensation", True))

    @property
    def initial_qpos(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in self.articulation_config["initial_qpos"].items()
        }

    @property
    def root_pose(self) -> sapien.Pose:
        config = self.articulation_config["root_pose"]
        return sapien.Pose(config["position"], config["quaternion_wxyz"])

    def validate_description_files(self):
        """Reject drift between the generated URDF and split source configs."""
        for path in (self.urdf_path, self.srdf_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing robot description file: {path}")

        expected_frequency = float(self.control_config["control_frequency"])
        if not np.isclose(expected_frequency * self.timestep, 1.0):
            raise ValueError(
                f"control_frequency ({expected_frequency}) and timestep ({self.timestep}) disagree"
            )

        root = ET.parse(self.urdf_path).getroot()
        urdf_joints = {
            joint.get("name"): joint
            for joint in root.findall("joint")
            if joint.get("type") != "fixed"
        }
        configured_limits = self.joint_limits_config["joints"]
        if set(urdf_joints) != set(configured_limits):
            raise ValueError(
                f"URDF/joint_limits mismatch; missing={sorted(set(urdf_joints)-set(configured_limits))}, "
                f"unknown={sorted(set(configured_limits)-set(urdf_joints))}"
            )
        if set(self.initial_qpos) != set(configured_limits):
            raise ValueError(
                f"initial_qpos/joint_limits mismatch; "
                f"missing={sorted(set(configured_limits)-set(self.initial_qpos))}, "
                f"unknown={sorted(set(self.initial_qpos)-set(configured_limits))}"
            )

        for name, expected in configured_limits.items():
            joint = urdf_joints[name]
            if joint.get("type") != expected["type"]:
                raise ValueError(
                    f"Joint type mismatch for {name}: URDF={joint.get('type')}, config={expected['type']}"
                )
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Active joint has no URDF limit: {name}")
            for key in ("lower", "upper", "velocity", "effort"):
                if not np.isclose(float(limit.get(key)), float(expected[key])):
                    raise ValueError(
                        f"Joint limit mismatch for {name}.{key}: "
                        f"URDF={limit.get(key)}, config={expected[key]}"
                    )

            initial = float(self.initial_qpos[name])
            if not float(expected["lower"]) <= initial <= float(expected["upper"]):
                raise ValueError(f"Initial qpos for {name} is outside configured limits: {initial}")

    def _make_physics_only_urdf(self) -> Path:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        for link in root.findall("link"):
            for visual in list(link.findall("visual")):
                link.remove(visual)
        for mesh in root.findall(".//collision/geometry/mesh"):
            filename = mesh.get("filename")
            if filename and not filename.startswith("package://"):
                mesh.set("filename", str((self.urdf_path.parent / filename).resolve()))
        fd, filename = tempfile.mkstemp(prefix="alohamini_physics_", suffix=".urdf", dir="/tmp")
        os.close(fd)
        path = Path(filename)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return path

    def load(self, scene: sapien.Scene, *, physics_only: bool = False, apply_srdf: bool = True):
        loader = scene.create_urdf_loader()
        loader.fix_root_link = bool(self.articulation_config.get("fix_root_link", True))
        load_path = self.urdf_path
        if physics_only:
            self._temporary_urdf = self._make_physics_only_urdf()
            load_path = self._temporary_urdf
        try:
            self.articulation = loader.load(
                str(load_path),
                srdf_file=str(self.srdf_path) if apply_srdf else None,
            )
        finally:
            if self._temporary_urdf is not None:
                self._temporary_urdf.unlink(missing_ok=True)
                self._temporary_urdf = None
        if self.articulation is None:
            raise RuntimeError(f"SAPIEN failed to load {self.urdf_path}")

        self.articulation.set_name("alohamini2pro")
        self.articulation.set_root_pose(self.root_pose)
        active_joints = self.articulation.get_active_joints()
        self.joint_names = [joint.get_name() for joint in active_joints]
        self.joints_by_name = {joint.get_name(): joint for joint in active_joints}
        self.validate_configuration()
        self.apply_mass_policy()
        self.configure_drives()
        self.set_joint_positions(self.initial_qpos, teleport=True)
        return self.articulation

    def validate_configuration(self):
        configured = set(self.initial_qpos)
        actual = set(self.joint_names)
        if configured != actual:
            missing = sorted(actual - configured)
            unknown = sorted(configured - actual)
            raise ValueError(f"initial_qpos mismatch; missing={missing}, unknown={unknown}")

        group_joints = []
        for group in self.control_config["joint_groups"].values():
            group_joints.extend(group["joints"])
        duplicates = sorted({name for name in group_joints if group_joints.count(name) > 1})
        if duplicates:
            raise ValueError(f"Joints occur in multiple drive groups: {duplicates}")
        if set(group_joints) != actual:
            raise ValueError(
                f"drive groups mismatch; missing={sorted(actual-set(group_joints))}, "
                f"unknown={sorted(set(group_joints)-actual)}"
            )

        limit_joints = set(self.joint_limits_config["joints"])
        if limit_joints != actual:
            raise ValueError(
                f"joint_limits mismatch; missing={sorted(actual-limit_joints)}, "
                f"unknown={sorted(limit_joints-actual)}"
            )

    def apply_mass_policy(self):
        policy = self.physical_config["mass_policy"]
        if policy.get("override_all_links", False):
            mass = float(policy["override_mass_kg"])
            for link in self.articulation.get_links():
                link.set_mass(mass)

    def configure_drives(self):
        for group in self.control_config["joint_groups"].values():
            stiffness = float(group["stiffness"])
            damping = float(group["damping"])
            default_force_limit = float(group["force_limit"])
            force_limit_by_joint = group.get("force_limit_by_joint", {})
            mode = group.get("drive_mode", "force")
            for name in group["joints"]:
                force_limit = float(force_limit_by_joint.get(name, default_force_limit))
                self.joints_by_name[name].set_drive_property(
                    stiffness=stiffness,
                    damping=damping,
                    force_limit=force_limit,
                    mode=mode,
                )

    def set_joint_positions(self, targets: dict[str, float], *, teleport: bool = False):
        unknown = sorted(set(targets) - set(self.joint_names))
        if unknown:
            raise ValueError(f"Unknown active joints: {unknown}")
        qpos = self.articulation.get_qpos().copy()
        for name, value in targets.items():
            index = self.joint_names.index(name)
            qpos[index] = float(value)
            self.joints_by_name[name].set_drive_target(float(value))
            self.joints_by_name[name].set_drive_velocity_target(0.0)
        if teleport:
            self.articulation.set_qpos(qpos)
            self.articulation.set_qvel(np.zeros_like(qpos))

    def gripper_command_to_joint(self, command: float) -> float:
        mapping = self.gripper_config["mapping"]
        command = float(np.clip(command, mapping["command_closed"], mapping["command_open"]))
        span = mapping["command_open"] - mapping["command_closed"]
        alpha = (command - mapping["command_closed"]) / span
        return float(mapping["joint_closed"] + alpha * (mapping["joint_open"] - mapping["joint_closed"]))

    def set_gripper_command(self, side: str, command: float, *, teleport: bool = False) -> float:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        joint_position = self.gripper_command_to_joint(command)
        self.set_joint_positions({f"{side}_gripper": joint_position}, teleport=teleport)
        return joint_position

    def apply_passive_force(self, enabled: bool | None = None):
        enabled = self.gravity_compensation if enabled is None else enabled
        if enabled:
            qf = self.articulation.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
            )
            self.articulation.set_qf(qf)
        else:
            self.articulation.set_qf(np.zeros_like(self.articulation.get_qf()))

    def step(self, scene: sapien.Scene, *, gravity_compensation: bool | None = None):
        self.apply_passive_force(gravity_compensation)
        scene.step()

    def target_errors(self, targets: dict[str, float] | None = None) -> list[dict]:
        targets = self.initial_qpos if targets is None else targets
        qpos = self.articulation.get_qpos()
        result = []
        for name, target in targets.items():
            actual = float(qpos[self.joint_names.index(name)])
            result.append(
                {"joint": name, "target": float(target), "actual": actual, "error": actual-float(target)}
            )
        return sorted(result, key=lambda item: abs(item["error"]), reverse=True)
