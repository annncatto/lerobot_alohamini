#!/usr/bin/env python3
"""Analyze an AlohaMini LeRobot evaluation dataset without connecting to a robot.

Example:
    cd /home/anncatto/lerobot_alohamini
    python scripts/analyze_alohamini_eval_dataset.py \
      --dataset.root /home/anncatto/.cache/huggingface/lerobot/local/eval_smolvla_visual_040000_2 \
      --training-dataset.root /home/anncatto/alohamini_gui/datasets/lerobot/local/alohamini_06_combined_63_visual_only \
      --policy.path /home/anncatto/lerobot_alohamini/outputs/smolvla_alohamini06_combined63_visual_only_dual_b32_100k_20260724/checkpoints/040000/pretrained_model

The analyzer reads the recorded Parquet data directly, compares commanded actions
with observed state, detects action-chunk discontinuities, optionally compares the
evaluation with its training dataset, and saves plots plus machine-readable reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pa_dataset

_matplotlib_cache = Path(tempfile.gettempdir()) / "lerobot-matplotlib-cache"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001


ACTION = "action"
OBS_STATE = "observation.state"
DEFAULT_OUTPUT_ROOT = Path("analysis_outputs")
EPS = 1e-8


@dataclass
class DatasetArrays:
    root: Path
    info: dict[str, Any]
    action: np.ndarray
    state: np.ndarray | None
    timestamp: np.ndarray
    frame_index: np.ndarray
    episode_index: np.ndarray
    action_names: list[str]
    state_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--training-dataset.root", dest="training_root", type=Path)
    parser.add_argument("--policy.path", dest="policy_path", type=Path)
    parser.add_argument(
        "--episode-duration-s",
        type=float,
        help="Known wall-clock duration of each episode. LeRobot recording timestamps are "
        "frame_index/fps and therefore cannot measure the real control frequency.",
    )
    parser.add_argument(
        "--action-chunk-size",
        default="auto",
        help="Executed actions per replan, or 'auto' to use metadata/detection.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-anomalies", type=int, default=12)
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def feature_names(info: dict[str, Any], key: str, width: int) -> list[str]:
    names = info.get("features", {}).get(key, {}).get("names")
    if isinstance(names, list) and len(names) == width:
        return [str(name) for name in names]
    return [f"{key}[{index}]" for index in range(width)]


def load_dataset_arrays(root: Path) -> DatasetArrays:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset is missing {info_path}")

    info = read_json(info_path)
    parquet_root = root / "data"
    dataset = pa_dataset.dataset(parquet_root, format="parquet")
    available = set(dataset.schema.names)
    required = {ACTION, "timestamp", "frame_index", "episode_index"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Dataset is missing required Parquet columns: {missing}")

    columns = [ACTION, "timestamp", "frame_index", "episode_index"]
    if OBS_STATE in available:
        columns.append(OBS_STATE)
    table = dataset.to_table(columns=columns)
    action = np.asarray(table[ACTION].to_pylist(), dtype=np.float64)
    state = (
        np.asarray(table[OBS_STATE].to_pylist(), dtype=np.float64) if OBS_STATE in columns else None
    )
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)
    frame_index = np.asarray(table["frame_index"], dtype=np.int64)
    episode_index = np.asarray(table["episode_index"], dtype=np.int64)
    return DatasetArrays(
        root=root,
        info=info,
        action=action,
        state=state,
        timestamp=timestamp,
        frame_index=frame_index,
        episode_index=episode_index,
        action_names=feature_names(info, ACTION, action.shape[1]),
        state_names=feature_names(info, OBS_STATE, state.shape[1]) if state is not None else [],
    )


def finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if finite.size else math.nan


def same_episode_delta(values: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.diff(values, axis=0)
    valid = np.diff(episodes) == 0
    return delta[valid], np.flatnonzero(valid)


def infer_chunk_size(
    action: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    *,
    minimum: int = 2,
    maximum: int = 256,
) -> tuple[int | None, float]:
    """Infer a likely replan period from excess action jumps at periodic boundaries.

    ACT-family policies commonly use 100-step chunks.  Keeping the search below
    that value can make a divisor such as 50 look like the fundamental period,
    because every 100-step boundary is also a 50-step boundary.
    """
    delta = np.abs(np.diff(action, axis=0))
    same_episode = np.diff(episode_index) == 0
    if not np.any(same_episode):
        return None, 1.0
    scale = np.nanquantile(delta[same_episode], 0.9, axis=0)
    active = np.isfinite(scale) & (scale > EPS)
    if not np.any(active):
        return None, 1.0
    normalized_jump = np.nanmedian(delta[:, active] / (scale[active] + EPS), axis=1)

    best_size: int | None = None
    best_score = 1.0
    upper = min(maximum, max(minimum, len(action) // 8))
    for size in range(minimum, upper + 1):
        boundary = same_episode & (((frame_index[:-1] + 1) % size) == 0)
        inside = same_episode & ~boundary
        if boundary.sum() < 5 or inside.sum() < 10:
            continue
        boundary_level = finite_quantile(normalized_jump[boundary], 0.75)
        inside_level = finite_quantile(normalized_jump[inside], 0.75)
        score = boundary_level / max(inside_level, EPS)
        # Prefer the shortest strong fundamental rather than its 2x/3x harmonics.
        if score > best_score * 1.05 or (
            score >= best_score * 0.95 and best_size is not None and size < best_size
        ):
            best_size = size
            best_score = score
    if best_score < 1.5:
        return None, best_score
    return best_size, best_score


def load_optional_config(path: Path | None, filename: str) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = path.expanduser().resolve() / filename
    return read_json(candidate) if candidate.is_file() else None


def resolve_chunk_size(
    requested: str,
    data: DatasetArrays,
    policy_path: Path | None,
) -> tuple[int | None, str, float]:
    if requested != "auto":
        size = int(requested)
        if size < 1:
            raise ValueError("--action-chunk-size must be positive")
        return size, "command line", 1.0

    evaluation_config = load_optional_config(data.root / "meta", "evaluation_config.json")
    if evaluation_config:
        size = evaluation_config.get("n_action_steps")
        if isinstance(size, int) and size > 0:
            return size, "meta/evaluation_config.json", 1.0

    detected_size, score = infer_chunk_size(data.action, data.frame_index, data.episode_index)
    if detected_size is not None:
        return detected_size, "periodic jump detection", score

    policy_config = load_optional_config(policy_path, "config.json")
    if policy_config:
        size = policy_config.get("n_action_steps")
        if isinstance(size, int) and size > 0:
            return size, "policy config.json", score
    return None, "unavailable", score


def load_training_reference(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    data = load_dataset_arrays(root)
    action_delta, _ = same_episode_delta(data.action, data.episode_index)
    return {
        "root": str(data.root),
        "data": data,
        "min": np.nanmin(data.action, axis=0),
        "max": np.nanmax(data.action, axis=0),
        "q01": np.nanquantile(data.action, 0.01, axis=0),
        "q99": np.nanquantile(data.action, 0.99, axis=0),
        "delta_q99": np.nanquantile(np.abs(action_delta), 0.99, axis=0),
        "delta_max": np.nanmax(np.abs(action_delta), axis=0),
    }


def training_range_tolerance(name: str, minimum: float, maximum: float) -> float:
    feature_range = max(0.0, maximum - minimum)
    if "lift_axis" in name:
        return max(2.0, feature_range * 0.01)
    if name == "theta.vel":
        return max(1.0, feature_range * 0.01)
    if name in {"x.vel", "y.vel"}:
        return max(0.01, feature_range * 0.01)
    if name.endswith(".pos"):
        return max(0.5, feature_range * 0.01)
    return max(1e-3, feature_range * 0.01)


def direction_changes(delta: np.ndarray, threshold: float) -> int:
    sign = np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0))
    sign = sign[sign != 0]
    return int(np.sum(sign[1:] != sign[:-1])) if sign.size > 1 else 0


def best_tracking_lag(action: np.ndarray, state: np.ndarray, max_lag: int = 50) -> tuple[int, float]:
    if len(action) < 4 or np.nanstd(action) < EPS or np.nanstd(state) < EPS:
        return 0, math.nan
    best_lag, best_corr = 0, -math.inf
    for lag in range(max_lag + 1):
        left = action[: len(action) - lag or None]
        right = state[lag:]
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.sum() < 3:
            continue
        corr = float(np.corrcoef(left[finite], right[finite])[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_lag, best_corr = lag, corr
    return best_lag, best_corr


def chunk_masks(data: DatasetArrays, chunk_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    same_episode = np.diff(data.episode_index) == 0
    if chunk_size is None:
        return np.zeros_like(same_episode), same_episode
    boundary = same_episode & (((data.frame_index[:-1] + 1) % chunk_size) == 0)
    return boundary, same_episode & ~boundary


def analyze_dimensions(
    data: DatasetArrays,
    training: dict[str, Any] | None,
    chunk_size: int | None,
) -> list[dict[str, Any]]:
    action_delta = np.diff(data.action, axis=0)
    state_delta = np.diff(data.state, axis=0) if data.state is not None else None
    same_episode = np.diff(data.episode_index) == 0
    boundary, inside = chunk_masks(data, chunk_size)
    fps = float(data.info.get("fps") or 1.0)
    state_index = {name: index for index, name in enumerate(data.state_names)}
    rows: list[dict[str, Any]] = []

    for index, name in enumerate(data.action_names):
        values = data.action[:, index]
        delta = action_delta[same_episode, index]
        abs_delta = np.abs(delta)
        threshold = max(finite_quantile(abs_delta, 0.5), EPS)
        row: dict[str, Any] = {
            "index": index,
            "name": name,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "q01": finite_quantile(values, 0.01),
            "q50": finite_quantile(values, 0.50),
            "q99": finite_quantile(values, 0.99),
            "delta_q90": finite_quantile(abs_delta, 0.90),
            "delta_q99": finite_quantile(abs_delta, 0.99),
            "delta_max": float(np.nanmax(abs_delta)) if abs_delta.size else 0.0,
            "direction_changes": direction_changes(delta, threshold),
            "still_fraction": float(np.mean(abs_delta <= max(threshold, 1e-6))) if abs_delta.size else 1.0,
        }
        if chunk_size is not None and np.any(boundary):
            boundary_delta = np.abs(action_delta[boundary, index])
            inside_delta = np.abs(action_delta[inside, index])
            boundary_mean = float(np.nanmean(boundary_delta))
            inside_mean = float(np.nanmean(inside_delta))
            row.update(
                {
                    "boundary_mean_delta": boundary_mean,
                    "boundary_q99_delta": finite_quantile(boundary_delta, 0.99),
                    "inside_mean_delta": inside_mean,
                    "inside_q99_delta": finite_quantile(inside_delta, 0.99),
                    "boundary_jump_ratio": boundary_mean / max(inside_mean, EPS),
                }
            )

        if training is not None and index < len(training["min"]):
            train_min = float(training["min"][index])
            train_max = float(training["max"][index])
            train_q01 = float(training["q01"][index])
            train_q99 = float(training["q99"][index])
            train_delta_q99 = float(training["delta_q99"][index])
            train_delta_max = float(training["delta_max"][index])
            tolerance = training_range_tolerance(name, train_min, train_max)
            row.update(
                {
                    "training_min": train_min,
                    "training_max": train_max,
                    "training_q01": train_q01,
                    "training_q99": train_q99,
                    "training_range_tolerance": tolerance,
                    "outside_training_range_fraction": float(
                        np.mean(
                            (values < train_min - tolerance)
                            | (values > train_max + tolerance)
                        )
                    ),
                    "outside_training_q01_q99_fraction": float(
                        np.mean((values < train_q01) | (values > train_q99))
                    ),
                    "training_delta_q99": train_delta_q99,
                    "training_delta_max": train_delta_max,
                    "delta_q99_training_max_ratio": row["delta_q99"]
                    / max(train_delta_max, tolerance, EPS),
                }
            )

        if data.state is not None and name in state_index:
            state_values = data.state[:, state_index[name]]
            error = values - state_values
            lag, correlation = best_tracking_lag(values, state_values, max_lag=max(1, int(fps * 2)))
            row.update(
                {
                    "tracking_mae": float(np.nanmean(np.abs(error))),
                    "tracking_rmse": float(np.sqrt(np.nanmean(error**2))),
                    "tracking_max_error": float(np.nanmax(np.abs(error))),
                    "tracking_lag_frames": lag,
                    "tracking_lag_seconds": lag / fps,
                    "tracking_correlation": correlation,
                    "state_direction_changes": direction_changes(
                        state_delta[same_episode, state_index[name]],
                        threshold,
                    ),
                }
            )
        rows.append(row)
    return rows


def integrity_report(
    data: DatasetArrays,
    episode_duration_s: float | None = None,
) -> dict[str, Any]:
    declared_frames = int(data.info.get("total_frames") or 0)
    dt = np.diff(data.timestamp)
    same_episode = np.diff(data.episode_index) == 0
    valid_dt = dt[same_episode]
    median_dt = finite_quantile(valid_dt, 0.5)
    expected_fps = float(data.info.get("fps") or 0.0)
    effective_fps = 1.0 / median_dt if median_dt > 0 else math.nan
    episode_count = int(len(np.unique(data.episode_index)))
    wall_clock_duration_s = (
        episode_duration_s * episode_count if episode_duration_s is not None else None
    )
    wall_clock_fps = (
        len(data.action) / wall_clock_duration_s
        if wall_clock_duration_s is not None and wall_clock_duration_s > 0
        else None
    )
    return {
        "declared_frames": declared_frames,
        "actual_frames": len(data.action),
        "frame_count_matches": declared_frames == len(data.action),
        "episodes": episode_count,
        "expected_fps": expected_fps,
        "effective_fps": effective_fps,
        "timestamp_fps": effective_fps,
        "timestamps_are_synthetic": True,
        "episode_duration_s": episode_duration_s,
        "wall_clock_duration_s": wall_clock_duration_s,
        "wall_clock_fps": wall_clock_fps,
        "timestamp_dt_median": median_dt,
        "timestamp_dt_q99": finite_quantile(valid_dt, 0.99),
        "non_monotonic_timestamps": int(np.sum(valid_dt <= 0)),
        "action_nan_or_inf": int(np.sum(~np.isfinite(data.action))),
        "state_nan_or_inf": int(np.sum(~np.isfinite(data.state))) if data.state is not None else None,
        "action_state_width_matches": (
            data.state is not None and data.action.shape[1] == data.state.shape[1]
        ),
        "action_state_names_match": data.action_names == data.state_names if data.state is not None else None,
    }


def select_anomalies(
    data: DatasetArrays,
    rows: list[dict[str, Any]],
    chunk_size: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    delta = np.abs(np.diff(data.action, axis=0))
    scale = np.array(
        [
            max(
                float(row.get("training_delta_max", row["delta_q99"])),
                float(row.get("training_range_tolerance", 0.0)),
                EPS,
            )
            for row in rows
        ]
    )
    score = np.nanmax(delta / scale, axis=1)
    dimension = np.nanargmax(delta / scale, axis=1)
    same_episode = np.diff(data.episode_index) == 0
    score[~same_episode] = -math.inf
    candidates = np.argsort(score)[::-1]
    selected: list[int] = []
    for candidate in candidates:
        if not np.isfinite(score[candidate]):
            continue
        if all(abs(int(candidate) - previous) >= 5 for previous in selected):
            selected.append(int(candidate))
        if len(selected) >= limit:
            break

    anomalies: list[dict[str, Any]] = []
    for index in selected:
        dim = int(dimension[index])
        anomalies.append(
            {
                "frame_before": int(data.frame_index[index]),
                "frame_after": int(data.frame_index[index + 1]),
                "episode_index": int(data.episode_index[index]),
                "timestamp": float(data.timestamp[index + 1]),
                "dimension": data.action_names[dim],
                "before": float(data.action[index, dim]),
                "after": float(data.action[index + 1, dim]),
                "absolute_delta": float(delta[index, dim]),
                "normalized_score": float(score[index]),
                "chunk_boundary": bool(
                    chunk_size is not None and (int(data.frame_index[index]) + 1) % chunk_size == 0
                ),
            }
        )
    return anomalies


def component_indices(names: list[str]) -> dict[str, list[int]]:
    groups = {
        "left_arm": [],
        "right_arm": [],
        "grippers": [],
        "mobile_base": [],
        "lift_axis": [],
    }
    for index, name in enumerate(names):
        if "gripper" in name:
            groups["grippers"].append(index)
        elif name.startswith("arm_left"):
            groups["left_arm"].append(index)
        elif name.startswith("arm_right"):
            groups["right_arm"].append(index)
        elif name in {"x.vel", "y.vel", "theta.vel"}:
            groups["mobile_base"].append(index)
        elif "lift_axis" in name:
            groups["lift_axis"].append(index)
    return groups


def plot_component(
    data: DatasetArrays,
    indices: list[int],
    title: str,
    output: Path,
    chunk_size: int | None,
) -> None:
    if not indices:
        return
    figure, axes = plt.subplots(len(indices), 1, figsize=(14, max(3, 2.4 * len(indices))), sharex=True)
    axes = np.atleast_1d(axes)
    state_index = {name: index for index, name in enumerate(data.state_names)}
    for axis, index in zip(axes, indices, strict=True):
        name = data.action_names[index]
        axis.plot(data.timestamp, data.action[:, index], label="action", linewidth=1.0)
        if data.state is not None and name in state_index:
            axis.plot(
                data.timestamp,
                data.state[:, state_index[name]],
                label="state",
                linewidth=0.9,
                alpha=0.8,
            )
        if chunk_size:
            boundary = np.flatnonzero((data.frame_index % chunk_size) == 0)
            for boundary_index in boundary:
                axis.axvline(data.timestamp[boundary_index], color="black", alpha=0.06, linewidth=0.5)
        axis.set_ylabel(name, fontsize=8)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Dataset time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_overview(
    data: DatasetArrays,
    rows: list[dict[str, Any]],
    output: Path,
    chunk_size: int | None,
) -> None:
    groups = component_indices(data.action_names)
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    dt = np.diff(data.timestamp)
    axes[0, 0].plot(data.timestamp[1:], dt * 1000, linewidth=0.8)
    axes[0, 0].axhline(1000.0 / float(data.info["fps"]), color="red", linestyle="--")
    axes[0, 0].set_title("Timestamp interval")
    axes[0, 0].set_ylabel("milliseconds")

    lift = groups["lift_axis"]
    if lift:
        index = lift[0]
        axes[0, 1].plot(data.timestamp, data.action[:, index], label="action")
        if data.state is not None and data.action_names[index] in data.state_names:
            state_index = data.state_names.index(data.action_names[index])
            axes[0, 1].plot(data.timestamp, data.state[:, state_index], label="state")
        axes[0, 1].legend()
    axes[0, 1].set_title("Lift axis")

    delta_q99 = np.array([row["delta_q99"] for row in rows])
    order = np.argsort(delta_q99)[-8:]
    axes[1, 0].barh([data.action_names[index] for index in order], delta_q99[order])
    axes[1, 0].set_title("Largest action delta q99")

    if chunk_size:
        ratios = np.array([row.get("boundary_jump_ratio", 0.0) for row in rows])
        order = np.argsort(ratios)[-8:]
        axes[1, 1].barh([data.action_names[index] for index in order], ratios[order])
        axes[1, 1].axvline(1.0, color="black", linestyle="--")
        axes[1, 1].set_title(f"Boundary/inside jump ratio (period={chunk_size})")
    else:
        axes[1, 1].text(0.5, 0.5, "Action chunk period unavailable", ha="center", va="center")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Dataset time (s)" if axis in axes[0] else "")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_training_comparison(
    data: DatasetArrays,
    training: dict[str, Any] | None,
    output: Path,
) -> None:
    if training is None:
        return
    groups = component_indices(data.action_names)
    selected = groups["lift_axis"] + groups["mobile_base"] + groups["grippers"]
    selected = selected[:6]
    if not selected:
        selected = list(range(min(6, data.action.shape[1])))
    figure, axes = plt.subplots(len(selected), 1, figsize=(12, max(3, 2.6 * len(selected))))
    axes = np.atleast_1d(axes)
    training_data: DatasetArrays = training["data"]
    for axis, index in zip(axes, selected, strict=True):
        axis.hist(
            training_data.action[:, index],
            bins=80,
            density=True,
            alpha=0.5,
            label="training",
        )
        axis.hist(data.action[:, index], bins=80, density=True, alpha=0.5, label="evaluation")
        axis.set_ylabel(data.action_names[index], fontsize=8)
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Training vs evaluation action distributions")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_frequency(data: DatasetArrays, output: Path) -> None:
    groups = component_indices(data.action_names)
    selected = groups["lift_axis"] + groups["mobile_base"]
    if not selected:
        return
    fps = float(data.info.get("fps") or 1.0)
    figure, axes = plt.subplots(len(selected), 1, figsize=(12, max(3, 2.5 * len(selected))))
    axes = np.atleast_1d(axes)
    for axis, index in zip(axes, selected, strict=True):
        values = data.action[:, index] - np.nanmean(data.action[:, index])
        spectrum = np.abs(np.fft.rfft(np.nan_to_num(values)))
        frequencies = np.fft.rfftfreq(len(values), d=1.0 / fps)
        axis.plot(frequencies[1:], spectrum[1:], linewidth=0.9)
        axis.set_ylabel(data.action_names[index], fontsize=8)
        axis.set_xlim(0, fps / 2)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Frequency (Hz)")
    figure.suptitle("Action frequency spectrum")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_chunk_boundaries(
    data: DatasetArrays,
    rows: list[dict[str, Any]],
    output: Path,
    chunk_size: int | None,
) -> None:
    if chunk_size is None:
        return
    valid = [row for row in rows if "boundary_q99_delta" in row]
    if not valid:
        return
    valid = sorted(valid, key=lambda row: row["boundary_jump_ratio"], reverse=True)[:12]
    positions = np.arange(len(valid))
    figure, axis = plt.subplots(figsize=(14, max(5, 0.45 * len(valid))))
    axis.barh(
        positions - 0.18,
        [row["inside_q99_delta"] for row in valid],
        height=0.36,
        label="inside chunk",
    )
    axis.barh(
        positions + 0.18,
        [row["boundary_q99_delta"] for row in valid],
        height=0.36,
        label="chunk boundary",
    )
    axis.set_yticks(positions, [row["name"] for row in valid])
    axis.invert_yaxis()
    axis.set_xlabel("Absolute action change (q99)")
    axis.set_title(f"Action discontinuity at replan boundaries (period={chunk_size})")
    axis.grid(axis="x", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_tracking_error(data: DatasetArrays, output: Path) -> None:
    if data.state is None:
        return
    state_index = {name: index for index, name in enumerate(data.state_names)}
    matched = [
        index for index, name in enumerate(data.action_names) if name in state_index
    ]
    if not matched:
        return
    errors = np.column_stack(
        [
            np.abs(data.action[:, index] - data.state[:, state_index[data.action_names[index]]])
            for index in matched
        ]
    )
    q99 = np.nanquantile(errors, 0.99, axis=0)
    selected = np.argsort(q99)[-8:]
    figure, axes = plt.subplots(
        len(selected),
        1,
        figsize=(14, max(4, 2.2 * len(selected))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, selected_index in zip(axes, selected, strict=True):
        action_index = matched[int(selected_index)]
        axis.plot(data.timestamp, errors[:, selected_index], linewidth=0.8)
        axis.set_ylabel(data.action_names[action_index], fontsize=8)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Dataset time (s)")
    figure.suptitle("Largest command/state tracking errors")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_anomaly_contact_sheet(
    data: DatasetArrays,
    anomalies: list[dict[str, Any]],
    output: Path,
) -> str | None:
    try:
        import av
        from PIL import Image, ImageDraw
    except ImportError:
        return "PyAV/Pillow is unavailable"

    camera_dirs = sorted((data.root / "videos").glob("observation.images.*"))
    if not camera_dirs or not anomalies:
        return "No video streams or anomalies"
    video_paths = []
    for camera_dir in camera_dirs[:2]:
        paths = sorted(camera_dir.glob("*/*.mp4"))
        if paths:
            video_paths.append((camera_dir.name, paths[0]))
    if not video_paths:
        return "No MP4 files found"

    selected_indices = [int(item["frame_after"]) for item in anomalies[:6]]
    rows: list[tuple[str, list[tuple[int, Any]]]] = []
    for camera_name, video_path in video_paths:
        wanted = set(selected_indices)
        decoded: dict[int, Any] = {}
        container = av.open(str(video_path))
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded[index] = frame.to_image().resize((400, 300))
            if index > max(wanted):
                break
        rows.append((camera_name, [(index, decoded.get(index)) for index in selected_indices]))

    cell_width, cell_height = 400, 330
    sheet = Image.new("RGB", (cell_width * len(selected_indices), cell_height * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (camera_name, frames) in enumerate(rows):
        for column, (frame_index, image) in enumerate(frames):
            x, y = column * cell_width, row_index * cell_height
            if image is not None:
                sheet.paste(image, (x, y + 25))
            draw.text((x + 5, y + 5), f"{camera_name} frame={frame_index}", fill="black")
    sheet.save(output)
    return None


def component_risk(rows: list[dict[str, Any]], indices: list[int]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for index in indices:
        row = rows[index]
        if row.get("boundary_jump_ratio", 0.0) >= 5:
            reasons.append(f"{row['name']} boundary jump ratio={row['boundary_jump_ratio']:.1f}")
        if row.get("delta_q99_training_max_ratio", 0.0) >= 2:
            reasons.append(
                f"{row['name']} delta q99 is "
                f"{row['delta_q99_training_max_ratio']:.1f}x training max step"
            )
        if row.get("outside_training_range_fraction", 0.0) >= 0.05:
            reasons.append(
                f"{row['name']} outside training range "
                f"{100 * row['outside_training_range_fraction']:.1f}%"
            )
    reasons = reasons[:8]
    if any("boundary" in reason or "delta q99" in reason for reason in reasons):
        return "danger", reasons
    if reasons:
        return "warning", reasons
    return "normal", reasons


def build_report(
    data: DatasetArrays,
    integrity: dict[str, Any],
    rows: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    chunk_size: int | None,
    chunk_source: str,
    chunk_score: float,
    training: dict[str, Any] | None,
    image_warning: str | None,
) -> dict[str, Any]:
    groups = component_indices(data.action_names)
    risks = {
        name: {"level": level, "reasons": reasons}
        for name, indices in groups.items()
        for level, reasons in [component_risk(rows, indices)]
    }
    rank = {"normal": 0, "warning": 1, "danger": 2}
    overall = max((item["level"] for item in risks.values()), key=rank.get, default="normal")
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(data.root),
        "training_dataset": training["root"] if training else None,
        "integrity": integrity,
        "action_chunk": {
            "size": chunk_size,
            "source": chunk_source,
            "detection_score": chunk_score,
        },
        "dimensions": rows,
        "anomalies": anomalies,
        "risk": {"overall": overall, "components": risks},
        "image_extraction_warning": image_warning,
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(report: dict[str, Any], output: Path) -> None:
    integrity = report["integrity"]
    chunk = report["action_chunk"]
    lines = [
        "# AlohaMini evaluation dataset analysis",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Training dataset: `{report['training_dataset'] or 'not provided'}`",
        f"- Frames: {integrity['actual_frames']} (declared {integrity['declared_frames']})",
        f"- Episodes: {integrity['episodes']}",
        f"- Dataset timeline FPS: expected {integrity['expected_fps']:.3f}, "
        f"timestamp-derived {integrity['timestamp_fps']:.3f} (synthetic frame_index/fps timeline)",
        (
            f"- Wall-clock FPS estimate: {integrity['wall_clock_fps']:.3f} "
            f"(known duration {integrity['wall_clock_duration_s']:.2f}s)"
            if integrity.get("wall_clock_fps") is not None
            else "- Wall-clock FPS estimate: unavailable (pass --episode-duration-s)"
        ),
        f"- Action chunk period: {chunk['size']} ({chunk['source']})",
        f"- Overall risk: **{report['risk']['overall']}**",
        "",
        "## Component risk",
        "",
        "| Component | Level | Reasons |",
        "|---|---|---|",
    ]
    for name, result in report["risk"]["components"].items():
        lines.append(f"| {name} | {result['level']} | {'; '.join(result['reasons']) or '-'} |")

    lines.extend(
        [
            "",
            "## Most discontinuous dimensions",
            "",
            "| Dimension | Δq99 | Boundary ratio | Training Δ ratio | Tracking MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    dimensions = sorted(report["dimensions"], key=lambda row: row["delta_q99"], reverse=True)[:12]
    for row in dimensions:
        lines.append(
            f"| {row['name']} | {row['delta_q99']:.4g} | "
            f"{row.get('boundary_jump_ratio', math.nan):.3g} | "
            f"{row.get('delta_q99_training_max_ratio', math.nan):.3g} | "
            f"{row.get('tracking_mae', math.nan):.3g} |"
        )

    lines.extend(
        [
            "",
            "## Top anomalies",
            "",
            "| Time | Frame | Dimension | Before | After | Δ | Chunk boundary |",
            "|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in report["anomalies"]:
        lines.append(
            f"| {item['timestamp']:.2f}s | {item['frame_after']} | {item['dimension']} | "
            f"{item['before']:.3g} | {item['after']:.3g} | {item['absolute_delta']:.3g} | "
            f"{item['chunk_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Generated plots",
            "",
            "- `overview.png`",
            "- `lift_axis.png`",
            "- `mobile_base.png`",
            "- `left_arm.png`",
            "- `right_arm.png`",
            "- `grippers.png`",
            "- `training_comparison.png`",
            "- `temporal_frequency.png`",
            "- `chunk_boundaries.png`",
            "- `action_state_tracking.png`",
            "- `anomaly_contact_sheet.png`",
            "",
            "Use `rerun_command.sh` for interactive inspection.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_anomalies < 1:
        raise ValueError("--top-anomalies must be positive")
    if args.episode_duration_s is not None and args.episode_duration_s <= 0:
        raise ValueError("--episode-duration-s must be positive")
    data = load_dataset_arrays(args.dataset_root)
    training = load_training_reference(args.training_root)
    chunk_size, chunk_source, chunk_score = resolve_chunk_size(
        args.action_chunk_size,
        data,
        args.policy_path,
    )
    integrity = integrity_report(data, args.episode_duration_s)
    rows = analyze_dimensions(data, training, chunk_size)
    anomalies = select_anomalies(data, rows, chunk_size, args.top_anomalies)

    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / DEFAULT_OUTPUT_ROOT / data.root.name).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)

    groups = component_indices(data.action_names)
    plot_overview(data, rows, output / "overview.png", chunk_size)
    for group_name, indices in groups.items():
        plot_component(
            data,
            indices,
            group_name.replace("_", " ").title(),
            output / f"{group_name}.png",
            chunk_size,
        )
    plot_training_comparison(data, training, output / "training_comparison.png")
    plot_frequency(data, output / "temporal_frequency.png")
    plot_chunk_boundaries(data, rows, output / "chunk_boundaries.png", chunk_size)
    plot_tracking_error(data, output / "action_state_tracking.png")
    image_warning = None
    if not args.no_images:
        image_warning = save_anomaly_contact_sheet(
            data,
            anomalies,
            output / "anomaly_contact_sheet.png",
        )

    report = build_report(
        data,
        integrity,
        rows,
        anomalies,
        chunk_size,
        chunk_source,
        chunk_score,
        training,
        image_warning,
    )
    (output / "report.json").write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_markdown(report, output / "report.md")
    save_csv(output / "summary.csv", rows)
    save_csv(output / "anomalies.csv", anomalies)
    command = (
        "# Interactive dataset viewer\n"
        "lerobot-dataset-viz \\\n"
        f"  --repo-id local/{data.root.name} \\\n"
        f"  --root {data.root} \\\n"
        "  --episode-index 0\n"
    )
    if args.policy_path:
        indices = " ".join(str(item["frame_after"]) for item in anomalies[:6])
        command += (
            "\n# Reuse the offline policy action diagnostic on anomalous frames\n"
            "python examples/alohamini/diagnose_policy_dataset_actions.py \\\n"
            f"  --policy.path {args.policy_path.expanduser().resolve()} \\\n"
            f"  --dataset.root {data.root} \\\n"
            f"  --dataset.repo_id local/{data.root.name} \\\n"
            f"  --indices {indices} \\\n"
            "  --device cuda \\\n"
            f"  --output {output / 'offline_policy_actions.json'}\n"
        )
    (output / "rerun_command.sh").write_text(command, encoding="utf-8")
    (output / "rerun_command.sh").chmod(0o755)
    print(output)


if __name__ == "__main__":
    main()
