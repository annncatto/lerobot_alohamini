#!/usr/bin/env python3
"""Drive AlohaMini from policy inference on a recorded dataset episode.

This is deliberately open loop: recorded images and recorded state are the
only policy observations.  Live robot observations are used once for an
initial-pose safety check, never as model input.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import torch

import lerobot.robots.alohamini  # noqa: F401 - register robot type
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.utils.action_quantization import snap_planar_velocity
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.robot_utils import precise_sleep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default=None)
    parser.add_argument("--dataset.episode", dest="episode", type=int, default=0)
    parser.add_argument("--robot.remote_ip", dest="remote_ip", default="127.0.0.1")
    parser.add_argument("--robot.id", dest="robot_id", default="my_alohamini")
    parser.add_argument(
        "--robot.robot_model",
        dest="robot_model",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
        default="alohamini2pro",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--action.base_snap_speed", dest="base_snap_speed", type=float, default=0.15)
    parser.add_argument("--action.base_snap_deadband", dest="base_snap_deadband", type=float, default=0.05)
    parser.add_argument(
        "--safety.max_initial_joint_error_deg",
        dest="max_initial_joint_error_deg",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--safety.max_initial_lift_error_mm",
        dest="max_initial_lift_error_mm",
        type=float,
        default=75.0,
    )
    return parser.parse_args()


def check_initial_pose(
    live_observation: dict,
    recorded_state: torch.Tensor,
    state_names: list[str],
    *,
    max_joint_error_deg: float,
    max_lift_error_mm: float,
) -> None:
    recorded = {name: float(recorded_state[index]) for index, name in enumerate(state_names)}
    joint_errors = {
        name: abs(float(live_observation[name]) - target)
        for name, target in recorded.items()
        if name.endswith(".pos")
    }
    lift_name = "lift_axis.height_mm"
    lift_error = abs(float(live_observation[lift_name]) - recorded[lift_name])
    worst_joint, worst_joint_error = max(joint_errors.items(), key=lambda item: item[1])
    print(
        f"Initial-pose check: worst_joint={worst_joint} error={worst_joint_error:.2f} deg, "
        f"lift_error={lift_error:.2f} mm",
        flush=True,
    )
    if worst_joint_error > max_joint_error_deg or lift_error > max_lift_error_mm:
        raise RuntimeError(
            "Robot does not match the recorded episode start pose: "
            f"{worst_joint} error={worst_joint_error:.2f} deg (limit={max_joint_error_deg:.2f}), "
            f"lift error={lift_error:.2f} mm (limit={max_lift_error_mm:.2f})."
        )


def main() -> None:
    args = parse_args()
    if args.episode < 0:
        raise ValueError("--dataset.episode cannot be negative")
    if args.base_snap_speed < 0 or args.base_snap_deadband < 0:
        raise ValueError("Base snap speed/deadband cannot be negative")

    dataset_root = args.dataset_root.expanduser().resolve()
    repo_id = args.dataset_repo_id or f"local/{dataset_root.name}"
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[args.episode],
        video_backend="pyav",
    )
    if not len(dataset):
        raise RuntimeError(f"Dataset episode {args.episode} is empty")

    action_names = dataset.meta.features["action"]["names"]
    state_names = dataset.meta.features["observation.state"]["names"]
    if not action_names or not state_names:
        raise RuntimeError("Dataset action/state feature names are required for safe replay")

    device = args.device or str(auto_select_torch_device())
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    if policy_cfg.type != "act":
        raise ValueError(f"Dataset replay currently supports ACT checkpoints, got {policy_cfg.type!r}")
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = device
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(args.policy_path, config=policy_cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    robot = AlohaMiniClient(
        AlohaMiniClientConfig(
            remote_ip=args.remote_ip,
            id=args.robot_id,
            robot_model=args.robot_model,
        )
    )
    expected_action_names = list(robot.action_features)
    if action_names != expected_action_names:
        raise RuntimeError(
            "Dataset/robot action ordering differs; refusing replay. "
            f"dataset={action_names}, robot={expected_action_names}"
        )

    robot.connect()
    actions_sent = 0
    overruns = 0
    episode_start = time.perf_counter()
    try:
        first_sample = dataset[0]
        # Safety-only read. This observation is never passed to the policy.
        live_observation = robot.get_observation()
        check_initial_pose(
            live_observation,
            first_sample["observation.state"],
            state_names,
            max_joint_error_deg=args.max_initial_joint_error_deg,
            max_lift_error_mm=args.max_initial_lift_error_mm,
        )

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        interval = 1.0 / dataset.fps
        next_tick = time.perf_counter()
        action_queue: deque[torch.Tensor] = deque()
        print(
            f"Starting open-loop replay: episode={args.episode}, frames={len(dataset)}, "
            f"fps={dataset.fps}, duration={len(dataset) / dataset.fps:.2f}s",
            flush=True,
        )
        for frame_index in range(len(dataset)):
            if not action_queue:
                # ACT consumes one observation per action chunk. Loading only these
                # dataset frames preserves ACT semantics and avoids decoding three
                # videos on ticks where the policy would only pop a cached action.
                sample = dataset[frame_index]
                observation = {key: sample[key] for key in policy_cfg.input_features}
                with torch.inference_mode():
                    normalized_chunk = policy.predict_action_chunk(preprocessor(observation))
                    physical_chunk = (
                        postprocessor(normalized_chunk)
                        .squeeze(0)[: policy_cfg.n_action_steps]
                        .detach()
                        .cpu()
                    )
                action_queue.extend(physical_chunk)

            physical_action = action_queue.popleft()
            action = {name: physical_action[index].item() for index, name in enumerate(action_names)}
            action = snap_planar_velocity(
                action,
                speed=args.base_snap_speed,
                deadband=args.base_snap_deadband,
            )
            robot.send_action(action)
            actions_sent += 1

            if frame_index % dataset.fps == 0:
                print(
                    f"Replay {frame_index / dataset.fps:6.1f}/{len(dataset) / dataset.fps:.1f}s "
                    f"gripper={action.get('arm_right_gripper.pos', 0.0):6.2f} "
                    f"x={action.get('x.vel', 0.0):+.2f} y={action.get('y.vel', 0.0):+.2f}",
                    flush=True,
                )

            next_tick += interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                precise_sleep(sleep_time)
            else:
                overruns += 1
                next_tick = time.perf_counter()
    finally:
        try:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        finally:
            robot.disconnect()

    elapsed = time.perf_counter() - episode_start
    print(
        f"Replay complete: actions={actions_sent}, elapsed={elapsed:.2f}s, "
        f"control_hz={actions_sent / elapsed:.2f}, overruns={overruns}",
        flush=True,
    )


if __name__ == "__main__":
    main()
