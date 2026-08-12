"""Bridge from alohamini_sim data-engine episodes to this repo's LeRobotDataset (v3.0).

The scripted sim engine (``engine.py`` / ``skills_runtime.py``) produces episodes shaped as::

    episode = {
        "steps": [
            {
                "qpos": float32[18],                  # sim joint state (writer_adapter.STATE_NAMES)
                "action": float32[18],                # controller targets (writer_adapter.ACTION_NAMES)
                "rgb": {cam_name: uint8[H, W, 3]},    # per-camera RGB, or None when rendering is off
            },
            ...
        ],
        "annotation": str,                            # natural-language task
        "command": {...}, "seed": int, "success": bool, "traces": [...], "final": {...},
    }

This module converts those episodes into a valid :class:`~lerobot.datasets.lerobot_dataset.
LeRobotDataset` using the in-repo v3.0 write API (``create`` -> ``add_frame`` -> ``save_episode``
-> ``finalize``), with feature names following the AlohaMini robot convention
(``AlohaMini._state_ft`` with the ``am-follower-6dof-hd`` arm profile, which matches the
maintained AlohaMini2Pro asset) so sim episodes can co-train with real recordings:

    observation.state / action: 18-D float32, names =
        [arm_left_{shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_yaw,wrist_roll,gripper}.pos,
         arm_right_{...}.pos, x.vel, y.vel, theta.vel, lift_axis.height_mm]
    observation.images.{cam}: uint8 (H, W, 3), video-encoded by default

Value mapping (documented per dimension, sim units unless noted):

- Arm joints: copied as-is (radians). Real recordings store normalized motor ranges; aligning
  units for co-training requires the real robot's calibration and is left to the caller.
- Gripper: sim finger-joint position (meters) by default. Pass ``gripper_range=(closed, open)``
  to map linearly onto the real robot's 0-100 gripper scale.
- x.vel / y.vel: body-frame base velocity in m/s. The sim records world-frame root positions,
  so the state velocity is the finite difference ``(p[t] - p[t-1]) * fps`` rotated into the base
  frame at yaw[t] (first frame = 0), and the action velocity is the one-control-step reaching
  velocity ``(target[t] - p[t]) * fps`` rotated the same way.
- theta.vel: base yaw rate in deg/s (same finite-difference/reaching-velocity scheme, wrapped).
- lift_axis.height_mm: sim ``vertical_move`` (meters) converted to millimeters.

Sim-only episode metadata (seed / command / traces / final) is not representable in the
LeRobotDataset schema and is dropped; keep the engine's state-only output if you need it.

CLI (episodes pickled as a list of episode dicts, or ``{"episodes": [...]}``)::

    python -m alohamini_sim.data_engine.lerobot_bridge \
        --episodes out/episodes.pkl --repo-id local/alohamini_sim_pick --root out/lerobot_ds

Requires this repo's uv env with the dataset extra: ``uv sync --locked --extra dataset``.
"""

from __future__ import annotations

import argparse
import pickle  # nosec B403 - CLI input is trusted local engine output
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.alohamini.model_specs import arm_state_keys
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features

from .data_gen.aspire_engine.writer_adapter import (
    ACTION_NAMES as SIM_ACTION_NAMES,
    STATE_NAMES as SIM_STATE_NAMES,
)

# The maintained asset is the HD 6-DoF follower arm plus one gripper joint.
SIM_ARM_PROFILE = "am-follower-6dof-hd"
BASE_FEATURE_NAMES = ("x.vel", "y.vel", "theta.vel", "lift_axis.height_mm")

# 18-D feature names, ordered exactly like AlohaMini._state_ft:
# left arm, right arm, base velocities, lift height.
ROBOT_STATE_NAMES: list[str] = [
    *arm_state_keys("arm_left", SIM_ARM_PROFILE),
    *arm_state_keys("arm_right", SIM_ARM_PROFILE),
    *BASE_FEATURE_NAMES,
]
LEGACY_ROBOT_STATE_NAMES: list[str] = [
    *arm_state_keys("arm_left", "so-arm-5dof"),
    *arm_state_keys("arm_right", "so-arm-5dof"),
    *BASE_FEATURE_NAMES,
]
LEGACY_STATE_NAMES = [
    "root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint", "vertical_move",
    "left_joint1", "right_joint1", "left_joint2", "right_joint2",
    "left_joint3", "right_joint3", "left_joint4", "right_joint4",
    "left_joint5", "right_joint5", "left_finger_joint1", "left_finger_joint2",
    "right_finger_joint1", "right_finger_joint2",
]
LEGACY_ACTION_NAMES = [
    "root_x_axis_joint_target", "root_y_axis_joint_target", "root_z_rotation_joint_target",
    "vertical_move_target", "left_joint1_target", "left_joint2_target", "left_joint3_target",
    "left_joint4_target", "left_joint5_target", "left_gripper_target", "right_joint1_target",
    "right_joint2_target", "right_joint3_target", "right_joint4_target", "right_joint5_target",
    "right_gripper_target",
]

DEFAULT_TASK = "alohamini_sim_task"

# ── Sim layout indices (resolved by name so they stay in sync with writer_adapter) ──

_QPOS_X = SIM_STATE_NAMES.index("root_x_axis_joint")
_QPOS_Y = SIM_STATE_NAMES.index("root_y_axis_joint")
_QPOS_YAW = SIM_STATE_NAMES.index("root_z_rotation_joint")
_QPOS_LIFT = SIM_STATE_NAMES.index("vertical_move")
_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)
_QPOS_LEFT_ARM = [SIM_STATE_NAMES.index(f"left_{name}") for name in _ARM_JOINTS]
_QPOS_LEFT_GRIP = SIM_STATE_NAMES.index("left_gripper")
_QPOS_RIGHT_ARM = [SIM_STATE_NAMES.index(f"right_{name}") for name in _ARM_JOINTS]
_QPOS_RIGHT_GRIP = SIM_STATE_NAMES.index("right_gripper")

_ACT_X = SIM_ACTION_NAMES.index("root_x_axis_joint_target")
_ACT_Y = SIM_ACTION_NAMES.index("root_y_axis_joint_target")
_ACT_YAW = SIM_ACTION_NAMES.index("root_z_rotation_joint_target")
_ACT_LIFT = SIM_ACTION_NAMES.index("vertical_move_target")
_ACT_LEFT_ARM = [SIM_ACTION_NAMES.index(f"left_{name}_target") for name in _ARM_JOINTS]
_ACT_LEFT_GRIP = SIM_ACTION_NAMES.index("left_gripper_target")
_ACT_RIGHT_ARM = [SIM_ACTION_NAMES.index(f"right_{name}_target") for name in _ARM_JOINTS]
_ACT_RIGHT_GRIP = SIM_ACTION_NAMES.index("right_gripper_target")


def episode_task(episode: dict[str, Any]) -> str:
    """Return the natural-language task string for an episode (same fallbacks as writer_adapter)."""
    return str(episode.get("annotation") or episode.get("command", {}).get("verb") or DEFAULT_TASK)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _world_to_body(vx_w: np.ndarray, vy_w: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate world-frame planar velocities into the base (body) frame at the given yaw."""
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return cos_yaw * vx_w + sin_yaw * vy_w, -sin_yaw * vx_w + cos_yaw * vy_w


def _map_gripper(values: np.ndarray, gripper_range: tuple[float, float] | None) -> np.ndarray:
    """Optionally map sim finger-joint positions (meters) onto the real 0-100 gripper scale."""
    if gripper_range is None:
        return values
    closed, opened = gripper_range
    if opened == closed:
        raise ValueError(f"gripper_range must have distinct endpoints, got {gripper_range}")
    return np.clip((values - closed) / (opened - closed) * 100.0, 0.0, 100.0)


def convert_state_action(
    qpos: np.ndarray,
    sim_action: np.ndarray,
    fps: int,
    gripper_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sim trajectories to the 18-D AlohaMini2Pro state/action matrices.

    Args:
        qpos: Sim joint states, float array of shape (T, 18) laid out as writer_adapter.STATE_NAMES.
        sim_action: Controller targets, shape (T, 18) laid out as writer_adapter.ACTION_NAMES.
        fps: Control frequency used to turn position differences into velocities.
        gripper_range: Optional (closed, open) finger-joint positions mapped onto 0-100.

    Returns:
        (state, action): two float32 arrays of shape (T, 18) ordered as ROBOT_STATE_NAMES.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    sim_action = np.asarray(sim_action, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 18:
        raise ValueError(f"qpos must be [T, 18], got {qpos.shape}")
    if sim_action.shape == (qpos.shape[0], len(LEGACY_ACTION_NAMES)):
        return _convert_legacy_state_action(qpos, sim_action, fps, gripper_range)
    if sim_action.shape != (qpos.shape[0], len(SIM_ACTION_NAMES)):
        raise ValueError(
            f"action must be [T, {len(SIM_ACTION_NAMES)}] (current) or "
            f"[T, {len(LEGACY_ACTION_NAMES)}] (legacy), got {sim_action.shape}"
        )

    num_steps = qpos.shape[0]
    state = np.zeros((num_steps, len(ROBOT_STATE_NAMES)), dtype=np.float64)
    action = np.zeros_like(state)
    yaw = qpos[:, _QPOS_YAW]

    # Arms: 6 joints + gripper per side, copied in real-robot feature order.
    state[:, 0:6] = qpos[:, _QPOS_LEFT_ARM]
    state[:, 6] = _map_gripper(qpos[:, _QPOS_LEFT_GRIP], gripper_range)
    state[:, 7:13] = qpos[:, _QPOS_RIGHT_ARM]
    state[:, 13] = _map_gripper(qpos[:, _QPOS_RIGHT_GRIP], gripper_range)
    action[:, 0:6] = sim_action[:, _ACT_LEFT_ARM]
    action[:, 6] = _map_gripper(sim_action[:, _ACT_LEFT_GRIP], gripper_range)
    action[:, 7:13] = sim_action[:, _ACT_RIGHT_ARM]
    action[:, 13] = _map_gripper(sim_action[:, _ACT_RIGHT_GRIP], gripper_range)

    # Base state: finite-difference world velocity rotated into the body frame (first frame = 0).
    vel_world = np.diff(qpos[:, [_QPOS_X, _QPOS_Y]], axis=0, prepend=qpos[:1, [_QPOS_X, _QPOS_Y]]) * fps
    state[:, 14], state[:, 15] = _world_to_body(vel_world[:, 0], vel_world[:, 1], yaw)
    dyaw = np.diff(yaw, prepend=yaw[:1])
    state[:, 16] = np.degrees(_wrap_angle(dyaw)) * fps
    state[:, 17] = qpos[:, _QPOS_LIFT] * 1000.0

    # Base action: one-control-step reaching velocity toward the commanded root targets.
    cmd_world = (sim_action[:, [_ACT_X, _ACT_Y]] - qpos[:, [_QPOS_X, _QPOS_Y]]) * fps
    action[:, 14], action[:, 15] = _world_to_body(cmd_world[:, 0], cmd_world[:, 1], yaw)
    action[:, 16] = np.degrees(_wrap_angle(sim_action[:, _ACT_YAW] - yaw)) * fps
    action[:, 17] = sim_action[:, _ACT_LIFT] * 1000.0

    return state.astype(np.float32), action.astype(np.float32)


def _convert_legacy_state_action(
    qpos: np.ndarray,
    sim_action: np.ndarray,
    fps: int,
    gripper_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read-only compatibility for episodes generated by the old 5-DoF Agent."""
    qi = {name: index for index, name in enumerate(LEGACY_STATE_NAMES)}
    ai = {name: index for index, name in enumerate(LEGACY_ACTION_NAMES)}
    state = np.zeros((qpos.shape[0], len(LEGACY_ROBOT_STATE_NAMES)), dtype=np.float64)
    action = np.zeros_like(state)
    left_q = [qi[f"left_joint{i}"] for i in range(1, 6)]
    right_q = [qi[f"right_joint{i}"] for i in range(1, 6)]
    left_a = [ai[f"left_joint{i}_target"] for i in range(1, 6)]
    right_a = [ai[f"right_joint{i}_target"] for i in range(1, 6)]
    state[:, 0:5] = qpos[:, left_q]
    state[:, 5] = _map_gripper(qpos[:, qi["left_finger_joint1"]], gripper_range)
    state[:, 6:11] = qpos[:, right_q]
    state[:, 11] = _map_gripper(qpos[:, qi["right_finger_joint1"]], gripper_range)
    action[:, 0:5] = sim_action[:, left_a]
    action[:, 5] = _map_gripper(sim_action[:, ai["left_gripper_target"]], gripper_range)
    action[:, 6:11] = sim_action[:, right_a]
    action[:, 11] = _map_gripper(sim_action[:, ai["right_gripper_target"]], gripper_range)
    yaw = qpos[:, qi["root_z_rotation_joint"]]
    xy = qpos[:, [qi["root_x_axis_joint"], qi["root_y_axis_joint"]]]
    vel_world = np.diff(xy, axis=0, prepend=xy[:1]) * fps
    state[:, 12], state[:, 13] = _world_to_body(vel_world[:, 0], vel_world[:, 1], yaw)
    state[:, 14] = np.degrees(_wrap_angle(np.diff(yaw, prepend=yaw[:1]))) * fps
    state[:, 15] = qpos[:, qi["vertical_move"]] * 1000.0
    cmd_xy = sim_action[:, [ai["root_x_axis_joint_target"], ai["root_y_axis_joint_target"]]]
    cmd_world = (cmd_xy - xy) * fps
    action[:, 12], action[:, 13] = _world_to_body(cmd_world[:, 0], cmd_world[:, 1], yaw)
    action[:, 14] = np.degrees(
        _wrap_angle(sim_action[:, ai["root_z_rotation_joint_target"]] - yaw)
    ) * fps
    action[:, 15] = sim_action[:, ai["vertical_move_target"]] * 1000.0
    return state.astype(np.float32), action.astype(np.float32)


def _episode_cameras(episode: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """Return {camera_name: (H, W, C)} from an episode's first step (empty when rendering is off)."""
    steps = episode.get("steps") or []
    if not steps:
        raise ValueError("cannot bridge an episode with no steps")
    rgb = steps[0].get("rgb")
    if rgb is None:
        return {}
    cameras: dict[str, tuple[int, int, int]] = {}
    for cam, img in rgb.items():
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError(
                f"camera '{cam}' must provide uint8 (H, W, 3) frames, got {img.dtype} {img.shape}"
            )
        cameras[str(cam)] = tuple(img.shape)
    return cameras


def build_features(
    cameras: dict[str, tuple[int, int, int]],
    use_videos: bool = True,
    state_names: list[str] | None = None,
) -> dict[str, dict]:
    """Build the LeRobotDataset features dict (same builder the recording pipeline uses)."""
    state_names = ROBOT_STATE_NAMES if state_names is None else state_names
    action_hw: dict[str, type | tuple] = dict.fromkeys(state_names, float)
    obs_hw: dict[str, type | tuple] = dict.fromkeys(state_names, float)
    obs_hw.update(cameras)
    return {
        **hw_to_dataset_features(action_hw, ACTION, use_videos),
        **hw_to_dataset_features(obs_hw, OBS_STR, use_videos),
    }


def _step_images(step: dict[str, Any], cameras: dict[str, tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """Validate and return the per-camera frames of one step against the expected camera set."""
    rgb = step.get("rgb") or {}
    if set(rgb) != set(cameras):
        raise ValueError(f"inconsistent cameras across steps: expected {sorted(cameras)}, got {sorted(rgb)}")
    images = {}
    for cam, shape in cameras.items():
        img = np.asarray(rgb[cam])
        if img.shape != shape or img.dtype != np.uint8:
            raise ValueError(f"camera '{cam}' frame drifted to {img.dtype} {img.shape}, expected {shape}")
        images[cam] = img
    return images


def write_episodes(
    episodes: list[dict[str, Any]],
    repo_id: str,
    root: str | Path,
    *,
    fps: int = 20,
    robot_type: str = "alohamini_sim",
    use_videos: bool = True,
    gripper_range: tuple[float, float] | None = None,
    parallel_encoding: bool = False,
) -> LeRobotDataset:
    """Write sim episodes into a new LeRobotDataset at ``root`` and return it (finalized).

    Args:
        episodes: List of engine episode dicts (see module docstring for the schema).
        repo_id: Dataset repo id, e.g. ``"local/alohamini_sim_pick"`` (not pushed to the Hub).
        root: Local directory to materialize the dataset into (must not already hold a dataset).
        fps: Control/render frequency of the sim episodes.
        robot_type: ``info.json`` robot type tag.
        use_videos: Encode cameras as MP4 (True, default) or store per-frame images (False).
        gripper_range: Optional (closed, open) sim finger-joint positions mapped onto 0-100.
        parallel_encoding: Encode multi-camera videos in a process pool. Off by default because
            it requires a ``__main__``-guarded entry point under spawn-based multiprocessing.
    """
    if not episodes:
        raise ValueError("no episodes to write")

    cameras = _episode_cameras(episodes[0])
    first_action = np.asarray((episodes[0].get("steps") or [])[0]["action"])
    if first_action.size == len(SIM_ACTION_NAMES):
        state_names = ROBOT_STATE_NAMES
    elif first_action.size == len(LEGACY_ACTION_NAMES):
        state_names = LEGACY_ROBOT_STATE_NAMES
    else:
        raise ValueError(f"unsupported sim action width: {first_action.size}")
    features = build_features(cameras, use_videos=use_videos, state_names=state_names)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=robot_type,
        use_videos=use_videos,
    )

    for episode in episodes:
        steps = episode.get("steps") or []
        if not steps:
            raise ValueError("cannot bridge an episode with no steps")
        if _episode_cameras(episode) != cameras:
            raise ValueError("all episodes must share the same camera set and resolutions")
        qpos = np.stack([step["qpos"] for step in steps])
        sim_action = np.stack([step["action"] for step in steps])
        state, action = convert_state_action(qpos, sim_action, fps, gripper_range=gripper_range)
        task = episode_task(episode)
        for index, step in enumerate(steps):
            frame: dict[str, Any] = {
                "observation.state": state[index],
                "action": action[index],
                "task": task,
            }
            for cam, img in _step_images(step, cameras).items():
                frame[f"{OBS_STR}.images.{cam}"] = img
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=parallel_encoding)

    dataset.finalize()
    return dataset


def load_episodes(path: str | Path) -> list[dict[str, Any]]:
    """Load engine episodes from a pickle file (a list of episodes or ``{"episodes": [...]}``)."""
    with Path(path).open("rb") as f:
        payload = pickle.load(f)  # nosec B301 - trusted local engine output
    if isinstance(payload, dict):
        payload = payload.get("episodes", payload)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"unsupported episodes payload of type {type(payload)}")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert alohamini_sim engine episodes into a LeRobotDataset (v3.0).",
    )
    parser.add_argument("--episodes", required=True, help="Pickle file with a list of engine episodes.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. local/alohamini_sim_pick.")
    parser.add_argument("--root", required=True, help="Output directory for the dataset.")
    parser.add_argument("--fps", type=int, default=20, help="Control/render frequency (default: 20).")
    parser.add_argument("--robot-type", default="alohamini_sim", help="info.json robot type tag.")
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Store per-frame images instead of encoding MP4 videos.",
    )
    parser.add_argument(
        "--gripper-range",
        type=float,
        nargs=2,
        metavar=("CLOSED", "OPEN"),
        default=None,
        help="Sim finger-joint positions (m) mapped linearly onto the real 0-100 gripper scale.",
    )
    parser.add_argument(
        "--parallel-encoding",
        action="store_true",
        help="Encode multi-camera videos in a process pool.",
    )
    args = parser.parse_args(argv)

    episodes = load_episodes(args.episodes)
    dataset = write_episodes(
        episodes,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        robot_type=args.robot_type,
        use_videos=not args.no_videos,
        gripper_range=tuple(args.gripper_range) if args.gripper_range else None,
        parallel_encoding=args.parallel_encoding,
    )
    print(
        f"Wrote {dataset.meta.total_episodes} episode(s), {dataset.meta.total_frames} frame(s) "
        f"to {dataset.root}"
    )


if __name__ == "__main__":
    main()
