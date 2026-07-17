#!/usr/bin/env python3
"""Replay a LeRobot evaluation dataset through ACT without connecting to hardware.

The report distinguishes perception/state failures from execution-horizon and
robot-command failures by retaining the complete predicted gripper chunk.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.device_utils import auto_select_torch_device, get_safe_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    parser.add_argument(
        "--dataset.repo_id",
        dest="dataset_repo_id",
        default=None,
        help="Defaults to the repo_id stored by the caller; only metadata loading needs a syntactically valid id.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for CSV/JSON/NPZ reports.")
    parser.add_argument("--stride", type=int, default=25, help="Analyze every Nth frame (25 = once/second at 25 Hz).")
    parser.add_argument("--max_frames", type=int, default=0, help="Maximum sampled frames; 0 analyzes all samples.")
    parser.add_argument("--device", default=None, help="Defaults to LeRobot automatic device selection.")
    parser.add_argument("--gripper_name", default="arm_right_gripper.pos")
    parser.add_argument("--open_threshold", type=float, default=30.0)
    parser.add_argument("--short_horizon", type=int, default=25)
    parser.add_argument("--video_backend", default="pyav")
    return parser.parse_args()


def first_crossing(values: np.ndarray, threshold: float) -> int:
    indices = np.flatnonzero(values > threshold)
    return int(indices[0]) if len(indices) else -1


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.max_frames < 0:
        raise ValueError("--max_frames cannot be negative")
    if args.short_horizon <= 0:
        raise ValueError("--short_horizon must be positive")

    dataset_root = args.dataset_root.expanduser().resolve()
    repo_id = args.dataset_repo_id or f"local/{dataset_root.name}"
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        video_backend=args.video_backend,
    )

    action_feature = dataset.meta.features["action"]
    action_names = action_feature.get("names")
    if not action_names or args.gripper_name not in action_names:
        raise ValueError(f"Action {args.gripper_name!r} not found in dataset action names: {action_names}")
    gripper_dim = action_names.index(args.gripper_name)
    x_dim = action_names.index("x.vel") if "x.vel" in action_names else None
    y_dim = action_names.index("y.vel") if "y.vel" in action_names else None

    requested_device = args.device or str(auto_select_torch_device())
    try:
        device = str(get_safe_torch_device(requested_device))
    except AssertionError as error:
        raise RuntimeError(f"Requested device {requested_device!r} is not available") from error
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    if policy_cfg.type != "act":
        raise ValueError(f"This diagnostic requires an ACT checkpoint, got {policy_cfg.type!r}")
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = device
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(args.policy_path, config=policy_cfg).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    input_keys = set(policy_cfg.input_features)
    sampled_indices = list(range(0, len(dataset), args.stride))
    if args.max_frames:
        sampled_indices = sampled_indices[: args.max_frames]

    rows: list[dict[str, float | int]] = []
    chunks: list[np.ndarray] = []
    for sample_number, frame_index in enumerate(sampled_indices, start=1):
        sample = dataset[frame_index]
        observation = {key: sample[key] for key in input_keys}
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        with torch.inference_mode():
            batch = preprocessor(observation)
            normalized_chunk = policy.predict_action_chunk(batch)
            physical_chunk = postprocessor(normalized_chunk).squeeze(0).detach().cpu().numpy()

        gripper = physical_chunk[:, gripper_dim]
        short_horizon = min(args.short_horizon, len(gripper))
        crossing = first_crossing(gripper, args.open_threshold)
        timestamp_value = sample.get("timestamp", frame_index / dataset.meta.fps)
        timestamp = float(timestamp_value.item() if hasattr(timestamp_value, "item") else timestamp_value)
        row: dict[str, float | int] = {
            "dataset_index": frame_index,
            "timestamp_s": timestamp,
            "gripper_step_0": float(gripper[0]),
            f"gripper_max_first_{args.short_horizon}": float(gripper[:short_horizon].max()),
            "gripper_max_full_chunk": float(gripper.max()),
            "first_step_above_threshold": crossing,
            "recorded_gripper_action": float(sample["action"][gripper_dim]),
        }
        if x_dim is not None:
            row["x_step_0"] = float(physical_chunk[0, x_dim])
            row["x_mean_full_chunk"] = float(physical_chunk[:, x_dim].mean())
        if y_dim is not None:
            row["y_step_0"] = float(physical_chunk[0, y_dim])
            row["y_mean_full_chunk"] = float(physical_chunk[:, y_dim].mean())
        rows.append(row)
        chunks.append(physical_chunk)
        print(
            f"[{sample_number:04d}/{len(sampled_indices):04d}] t={timestamp:7.2f}s "
            f"gripper first={gripper[0]:6.2f} first{short_horizon}_max={gripper[:short_horizon].max():6.2f} "
            f"full_max={gripper.max():6.2f} crossing={crossing}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "gripper_chunk_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    chunk_array = np.stack(chunks) if chunks else np.empty((0, 0, len(action_names)), dtype=np.float32)
    np.savez_compressed(
        args.output / "predicted_action_chunks.npz",
        dataset_indices=np.asarray(sampled_indices),
        action_names=np.asarray(action_names),
        chunks=chunk_array,
    )

    full_crossings = sum(row["first_step_above_threshold"] >= 0 for row in rows)
    short_crossings = sum(
        0 <= row["first_step_above_threshold"] < args.short_horizon for row in rows
    )
    late_only = full_crossings - short_crossings
    summary = {
        "policy_path": str(Path(args.policy_path).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "device": device,
        "sampled_frames": len(rows),
        "dataset_frames": len(dataset),
        "stride": args.stride,
        "gripper_name": args.gripper_name,
        "gripper_dim": gripper_dim,
        "open_threshold": args.open_threshold,
        "short_horizon": args.short_horizon,
        "frames_with_open_in_full_chunk": full_crossings,
        "frames_with_open_in_first_horizon": short_crossings,
        "frames_with_open_only_after_horizon": late_only,
        "max_predicted_gripper": max((row["gripper_max_full_chunk"] for row in rows), default=None),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
