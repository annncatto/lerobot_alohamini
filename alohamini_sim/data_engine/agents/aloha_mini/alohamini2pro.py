"""ManiSkill Agent backed by the maintained RoboTwin AlohaMini2Pro asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import sapien
import torch
import yaml

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDBaseVelControllerConfig,
    PDJointPosController,
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


ASSET_DIR = Path(__file__).resolve().parent / "assets/alohamini2pro"
BASE_JOINTS = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]
LIFT_JOINTS = ["vertical_move"]
LEFT_ARM_JOINTS = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_yaw_joint",
    "left_wrist_roll",
]
RIGHT_ARM_JOINTS = [name.replace("left_", "right_") for name in LEFT_ARM_JOINTS]
LEFT_GRIPPER_JOINTS = ["left_gripper"]
RIGHT_GRIPPER_JOINTS = ["right_gripper"]

Q_OPEN = -1.8030294104
Q_CLOSED = 0.32
ARM_HOME = [0.0, 0.0735978469, 0.0169070196, 1.3960929593, 0.0, 0.0]
ACTIVE_JOINT_ORDER = (
    BASE_JOINTS
    + LIFT_JOINTS
    + [name for pair in zip(LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, strict=True) for name in pair]
    + LEFT_GRIPPER_JOINTS
    + RIGHT_GRIPPER_JOINTS
)
_REST_BY_JOINT = dict(zip(LEFT_ARM_JOINTS, ARM_HOME, strict=True))
_REST_BY_JOINT.update(dict(zip(RIGHT_ARM_JOINTS, ARM_HOME, strict=True)))
_REST_BY_JOINT.update(
    root_x_axis_joint=0.0,
    root_y_axis_joint=0.0,
    root_z_rotation_joint=0.0,
    vertical_move=0.20,
    left_gripper=Q_OPEN,
    right_gripper=Q_OPEN,
)
REST_QPOS = np.asarray([_REST_BY_JOINT[name] for name in ACTIVE_JOINT_ORDER], dtype=np.float32)

with (ASSET_DIR / "config.yml").open(encoding="utf-8") as stream:
    RUNTIME_CONFIG = yaml.safe_load(stream)
DELTA_MATRIX = np.asarray(RUNTIME_CONFIG["delta_matrix"], dtype=np.float32)
TCP_IN_TOOL = np.asarray(
    [RUNTIME_CONFIG["gripper_bias"], 0.0, 0.0], dtype=np.float32
) - np.asarray(RUNTIME_CONFIG["gripper_center_offset"], dtype=np.float32)
TCP_IN_FIXED_JAW = DELTA_MATRIX @ TCP_IN_TOOL

with (ASSET_DIR / "config/actuator_parameters.yaml").open(encoding="utf-8") as stream:
    ACTUATOR_CONFIG = yaml.safe_load(stream)
_SERVOS = ACTUATOR_CONFIG["servos"]
_ARM_SERVO_MAP = ACTUATOR_CONFIG["joint_mapping"]["arms"]
ARM_FORCE_LIMITS = [
    _SERVOS[_ARM_SERVO_MAP[name.removeprefix("left_")]]["simulation_force_limit_nm"]
    for name in LEFT_ARM_JOINTS
]
ARM_VELOCITY_LIMITS = [
    _SERVOS[_ARM_SERVO_MAP[name.removeprefix("left_")]]["simulation_velocity_limit_rad_s"]
    for name in LEFT_ARM_JOINTS
]
GRIPPER_FORCE_LIMIT = _SERVOS["sts3250"]["simulation_force_limit_nm"]
GRIPPER_VELOCITY_LIMIT = _SERVOS["sts3250"]["simulation_velocity_limit_rad_s"]
LIFT_FORCE_LIMIT = ACTUATOR_CONFIG["joint_mapping"]["lift"]["simulation_force_limit_n"]
LIFT_VELOCITY_LIMIT = ACTUATOR_CONFIG["joint_mapping"]["lift"]["simulation_velocity_limit_m_s"]
CAMERA_LINK_TO_SAPIEN_Q = [0.5, 0.5, -0.5, 0.5]
CAMERA_INTRINSIC_PROVISIONAL = np.asarray(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
CAMERA_LINKS = {
    "front_camera": "front_camera",
    "back_camera": "back_camera",
    "chest_camera": "chest_camera",
    "left_wrist_camera": "left_camera",
    "right_wrist_camera": "right_camera",
}
with (ASSET_DIR / "config/hardware_joint_map_right.yaml").open(encoding="utf-8") as stream:
    _HARDWARE_JOINTS = yaml.safe_load(stream)["joints"]
_HARDWARE_KEYS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)
SAFE_ARM_LOWER = [
    max(-np.pi, float(_HARDWARE_JOINTS[name].get("safe_q_min_rad", -np.pi)))
    for name in _HARDWARE_KEYS
]
SAFE_ARM_UPPER = [
    min(np.pi, float(_HARDWARE_JOINTS[name].get("safe_q_max_rad", np.pi)))
    for name in _HARDWARE_KEYS
]


class RateLimitedPDJointPosController(PDJointPosController):
    """Clamp target changes to the actuator limit once per control step."""

    config: "RateLimitedPDJointPosControllerConfig"

    def set_action(self, action: torch.Tensor):
        requested = self._preprocess_action(action)
        max_delta = torch.as_tensor(
            np.array(np.broadcast_to(self.config.velocity_limit, len(self.joints)), copy=True),
            dtype=requested.dtype,
            device=requested.device,
        ) / float(self.control_freq)
        self._start_qpos = self.qpos
        self._target_qpos = torch.clamp(
            requested,
            min=self._target_qpos - max_delta,
            max=self._target_qpos + max_delta,
        )
        self.set_drive_targets(self._target_qpos)


@dataclass
class RateLimitedPDJointPosControllerConfig(PDJointPosControllerConfig):
    velocity_limit: float | Sequence[float] = float("inf")
    controller_cls = RateLimitedPDJointPosController


@register_agent()
class AlohaMini2ProSim(BaseAgent):
    """The maintained 6-DoF-per-arm AlohaMini2Pro simulator embodiment."""

    uid = "alohamini2pro_sim"
    urdf_path = str(ASSET_DIR / "urdf/alohamini2pro.urdf")
    srdf_path = str(ASSET_DIR / "urdf/alohamini2pro.srdf")
    fix_root_link = True
    load_multiple_collisions = False
    disable_self_collisions = False

    Q_OPEN = Q_OPEN
    Q_CLOSED = Q_CLOSED
    DELTA_MATRIX = DELTA_MATRIX
    TCP_IN_FIXED_JAW = TCP_IN_FIXED_JAW
    safe_arm_lower = SAFE_ARM_LOWER
    safe_arm_upper = SAFE_ARM_UPPER
    base_joint_names = BASE_JOINTS
    lift_joint_names = LIFT_JOINTS
    left_arm_joint_names = LEFT_ARM_JOINTS
    right_arm_joint_names = RIGHT_ARM_JOINTS
    left_gripper_joint_names = LEFT_GRIPPER_JOINTS
    right_gripper_joint_names = RIGHT_GRIPPER_JOINTS

    @property
    def _sensor_configs(self):
        """Five URDF-link-mounted cameras; real intrinsics/extrinsics are provisional."""
        optical_pose = sapien.Pose(q=CAMERA_LINK_TO_SAPIEN_Q)
        return [
            CameraConfig(
                uid=uid,
                pose=optical_pose,
                width=640,
                height=480,
                intrinsic=CAMERA_INTRINSIC_PROVISIONAL.copy(),
                near=0.01,
                far=10.0,
                entity_uid=link_name,
            )
            for uid, link_name in CAMERA_LINKS.items()
        ]

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=1.5, dynamic_friction=1.2, restitution=0.0)
        ),
        link={
            name: dict(material="gripper", patch_radius=0.02, min_patch_radius=0.005)
            for name in (
                "left_Fixed_Jaw",
                "left_Moving_Jaw",
                "right_Fixed_Jaw",
                "right_Moving_Jaw",
            )
        },
    )
    keyframes = dict(
        rest=Keyframe(qpos=REST_QPOS, pose=sapien.Pose()),
        ready=Keyframe(qpos=REST_QPOS, pose=sapien.Pose()),
    )

    def _position_groups(self, fixed_base: bool) -> dict:
        if fixed_base:
            base = PDJointPosControllerConfig(
                self.base_joint_names,
                lower=None,
                upper=None,
                stiffness=1000.0,
                damping=200.0,
                force_limit=500.0,
                normalize_action=False,
            )
        else:
            base = PDBaseVelControllerConfig(
                self.base_joint_names,
                lower=[-0.15, -0.15, -np.pi / 4],
                upper=[0.15, 0.15, np.pi / 4],
                damping=200.0,
                force_limit=500.0,
                normalize_action=False,
            )
        common = dict(normalize_action=False, interpolate=False)
        lift = RateLimitedPDJointPosControllerConfig(
            self.lift_joint_names,
            lower=[-0.3],
            upper=[0.3],
            stiffness=1000.0,
            damping=200.0,
            force_limit=LIFT_FORCE_LIMIT,
            velocity_limit=LIFT_VELOCITY_LIMIT,
            **common,
        )
        left_arm = RateLimitedPDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=SAFE_ARM_LOWER,
            upper=SAFE_ARM_UPPER,
            stiffness=1000.0,
            damping=200.0,
            force_limit=ARM_FORCE_LIMITS,
            velocity_limit=ARM_VELOCITY_LIMITS,
            **common,
        )
        right_arm = RateLimitedPDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=SAFE_ARM_LOWER,
            upper=SAFE_ARM_UPPER,
            stiffness=1000.0,
            damping=200.0,
            force_limit=ARM_FORCE_LIMITS,
            velocity_limit=ARM_VELOCITY_LIMITS,
            **common,
        )
        left_gripper = RateLimitedPDJointPosControllerConfig(
            self.left_gripper_joint_names,
            lower=[Q_OPEN],
            upper=[Q_CLOSED],
            stiffness=1000.0,
            damping=200.0,
            force_limit=GRIPPER_FORCE_LIMIT,
            velocity_limit=GRIPPER_VELOCITY_LIMIT,
            **common,
        )
        right_gripper = RateLimitedPDJointPosControllerConfig(
            self.right_gripper_joint_names,
            lower=[Q_OPEN],
            upper=[Q_CLOSED],
            stiffness=1000.0,
            damping=200.0,
            force_limit=GRIPPER_FORCE_LIMIT,
            velocity_limit=GRIPPER_VELOCITY_LIMIT,
            **common,
        )
        return dict(
            base=base,
            lift=lift,
            left_arm=left_arm,
            left_gripper=left_gripper,
            right_arm=right_arm,
            right_gripper=right_gripper,
        )

    @property
    def _controller_configs(self):
        return deepcopy_dict(
            dict(
                pd_joint_pos=self._position_groups(fixed_base=False),
                pd_joint_pos_fixed_base=self._position_groups(fixed_base=True),
            )
        )

    def _after_init(self):
        self.base_link = self.robot.links_map["base_link"]
        self.left_fixed_jaw = self.robot.links_map["left_Fixed_Jaw"]
        self.left_moving_jaw = self.robot.links_map["left_Moving_Jaw"]
        self.right_fixed_jaw = self.robot.links_map["right_Fixed_Jaw"]
        self.right_moving_jaw = self.robot.links_map["right_Moving_Jaw"]

        # Compatibility names consumed by the existing data-engine IK helpers.
        self.left_ee_link = self.left_fixed_jaw
        self.right_ee_link = self.right_fixed_jaw
        self.left_palm_link = self.left_fixed_jaw
        self.right_palm_link = self.right_fixed_jaw
        self.left_finger1_tip = self.left_fixed_jaw
        self.left_finger2_tip = self.left_moving_jaw
        self.right_finger1_tip = self.right_fixed_jaw
        self.right_finger2_tip = self.right_moving_jaw

    @property
    def left_tcp_pose(self) -> Pose:
        return self.left_fixed_jaw.pose * Pose.create_from_pq(TCP_IN_FIXED_JAW)

    @property
    def right_tcp_pose(self) -> Pose:
        return self.right_fixed_jaw.pose * Pose.create_from_pq(TCP_IN_FIXED_JAW)

    @property
    def tcp_pose(self) -> Pose:
        return self.left_tcp_pose

    @property
    def tcp_pos(self):
        return self.left_tcp_pose.p

    @property
    def tcp_pos_2(self):
        return self.right_tcp_pose.p

    def get_left_ee_pose(self):
        return self.left_tcp_pose

    def get_right_ee_pose(self):
        return self.right_tcp_pose

    def is_grasping(
        self,
        object: Actor,
        side: str | None = None,
        min_force: float = 0.5,
        arm_id: int | None = None,
        **_: object,
    ):
        if side is None:
            side = "left" if arm_id in (None, 1) else "right"
        fixed = self.left_fixed_jaw if side == "left" else self.right_fixed_jaw
        moving = self.left_moving_jaw if side == "left" else self.right_moving_jaw
        fixed_force = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(fixed, object), dim=1
        )
        moving_force = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(moving, object), dim=1
        )
        return (fixed_force >= min_force) & (moving_force >= min_force)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        qvel = self.robot.get_qvel()
        return torch.all(torch.abs(qvel[:, 3:]) <= threshold, dim=1) & torch.all(
            torch.abs(qvel[:, :3]) <= base_threshold, dim=1
        )
