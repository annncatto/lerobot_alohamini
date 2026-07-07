#!/usr/bin/env python3
"""Episode 加权采样器。

用途：
- 不删除早期或质量稍差的 episode，只降低它们被抽到的概率。
- 可按 episode 长度自动修正权重：长 episode 稍降权，短 episode 稍升权。
- 可按 frame 动作自动加权：底盘横/纵移动、夹爪打开、放置阶段更常被抽到。

采样方式：
1. 先按 episode 权重随机抽一个 episode。
2. 再在这个 episode 内按 frame 权重抽一个 frame；没有 frame 权重时退回均匀随机。

常用参数：
- first_n_episodes / first_n_weight：给前 N 个 episode 调权；first_n_episodes=0 表示关闭。
- length_weight_power：开启长度修正；0 表示关闭，0.5 比较温和。
- min/max_length_multiplier：限制长度修正倍率，避免极端长短 episode 影响过大。
- base_motion_multiplier：含有 |x.vel| 或 |y.vel| 的 frame 加权。
- gripper_open_multiplier：夹爪向打开方向运动的 frame 加权。
- placement_multiplier：每个 episode 末尾放置阶段 frame 加权。
"""

import logging
from collections.abc import Iterator
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


class WeightedEpisodeSampler:
    """Sample training frames through weighted episode selection.

    Each draw first chooses an episode according to the configured episode weight,
    then chooses one valid frame uniformly inside that episode. This keeps useful
    early demonstrations in the training set while reducing how often they appear.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        first_n_episodes: int = 0,
        first_n_weight: float = 1.0,
        later_weight: float = 1.0,
        length_weight_power: float = 0.0,
        min_length_multiplier: float = 0.5,
        max_length_multiplier: float = 2.0,
        num_samples_per_epoch: int | None = None,
        dataset_getter: Callable[[], object | None] | None = None,
        base_motion_multiplier: float = 2.0,
        base_motion_threshold: float = 1e-4,
        gripper_open_multiplier: float = 2.0,
        gripper_delta_threshold: float = 1e-3,
        placement_multiplier: float = 1.5,
        placement_last_fraction: float = 0.2,
        max_frame_multiplier: float = 6.0,
    ):
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")
        if first_n_episodes < 0:
            raise ValueError(f"first_n_episodes must be >= 0, got {first_n_episodes}")
        if first_n_weight < 0 or later_weight < 0:
            raise ValueError("Episode weights must be non-negative.")
        if first_n_weight == 0 and later_weight == 0:
            raise ValueError("At least one episode weight must be > 0.")
        if length_weight_power < 0:
            raise ValueError(f"length_weight_power must be >= 0, got {length_weight_power}")
        if min_length_multiplier <= 0 or max_length_multiplier <= 0:
            raise ValueError("Length multipliers must be > 0.")
        if min_length_multiplier > max_length_multiplier:
            raise ValueError(
                "min_length_multiplier must be <= max_length_multiplier, "
                f"got {min_length_multiplier} > {max_length_multiplier}"
            )
        for name, value in {
            "base_motion_multiplier": base_motion_multiplier,
            "gripper_open_multiplier": gripper_open_multiplier,
            "placement_multiplier": placement_multiplier,
            "max_frame_multiplier": max_frame_multiplier,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if base_motion_threshold < 0:
            raise ValueError(f"base_motion_threshold must be >= 0, got {base_motion_threshold}")
        if gripper_delta_threshold < 0:
            raise ValueError(f"gripper_delta_threshold must be >= 0, got {gripper_delta_threshold}")
        if not 0 <= placement_last_fraction <= 1:
            raise ValueError(f"placement_last_fraction must be in [0, 1], got {placement_last_fraction}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                "dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        # 只保留当前 dataset 配置允许使用的 episode。
        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        # drop_n_* 会移除每个 episode 开头/结尾不适合训练的 frame。
        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has no valid frames after dropping first/last frames. Skipping.",
                episode_idx,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError("No valid frames remain after applying episode filters and frame drops.")

        self._episode_indices = np.arange(len(from_indices), dtype=np.int64)[used]
        self._starts = starts[used]
        self._lengths = lengths[used]
        self._num_samples = int(num_samples_per_epoch or self._lengths.sum())
        if self._num_samples <= 0:
            raise ValueError(f"num_samples_per_epoch must be > 0, got {self._num_samples}")

        # 基础权重：first_n_episodes=0 时不区分早期/后期 episode。
        weights = np.where(self._episode_indices < first_n_episodes, first_n_weight, later_weight)
        weights = weights.astype(np.float64)
        length_multipliers = np.ones_like(weights)
        if length_weight_power > 0:
            # 长度修正：长度超过中位数的 episode 会被稍微降权，短 episode 会被稍微升权。
            median_length = float(np.median(self._lengths))
            length_multipliers = (median_length / self._lengths.astype(np.float64)) ** length_weight_power
            length_multipliers = np.clip(length_multipliers, min_length_multiplier, max_length_multiplier)
            weights *= length_multipliers

        valid_weight_mask = weights > 0
        if not valid_weight_mask.any():
            raise ValueError("All selected episodes have zero sampling weight.")

        self._episode_indices = self._episode_indices[valid_weight_mask]
        self._starts = self._starts[valid_weight_mask]
        self._lengths = self._lengths[valid_weight_mask]
        length_multipliers = length_multipliers[valid_weight_mask]
        weights = weights[valid_weight_mask]
        self._episode_probs = torch.as_tensor(weights / weights.sum(), dtype=torch.double)
        self._frame_cdfs = self._build_frame_cdfs(
            dataset_getter=dataset_getter,
            base_motion_multiplier=base_motion_multiplier,
            base_motion_threshold=base_motion_threshold,
            gripper_open_multiplier=gripper_open_multiplier,
            gripper_delta_threshold=gripper_delta_threshold,
            placement_multiplier=placement_multiplier,
            placement_last_fraction=placement_last_fraction,
            max_frame_multiplier=max_frame_multiplier,
        )

        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

        logger.info(
            "WeightedEpisodeSampler: %d episodes, %d samples/epoch, first_n=%d, "
            "first_weight=%.3f, later_weight=%.3f, length_power=%.3f, "
            "length_multiplier_range=[%.3f, %.3f]",
            len(self._episode_indices),
            self._num_samples,
            first_n_episodes,
            first_n_weight,
            later_weight,
            length_weight_power,
            float(length_multipliers.min()),
            float(length_multipliers.max()),
        )

    def _build_frame_cdfs(
        self,
        dataset_getter: Callable[[], object | None] | None,
        base_motion_multiplier: float,
        base_motion_threshold: float,
        gripper_open_multiplier: float,
        gripper_delta_threshold: float,
        placement_multiplier: float,
        placement_last_fraction: float,
        max_frame_multiplier: float,
    ) -> list[np.ndarray] | None:
        """Build per-episode CDFs for frame-level weighted sampling.

        这里不改 ACT loss，只改变 frame 被采样到的概率：
        - 底盘 x/y 速度明显不为 0 的 frame 更常出现。
        - 夹爪向打开方向变化的 frame 更常出现。
        - episode 末尾默认视作放置阶段，更常出现。
        """
        if dataset_getter is None:
            return None
        dataset = dataset_getter()
        if dataset is None:
            logger.warning("Frame weighting requested but dataset is not available. Using uniform frames.")
            return None

        action_names = self._get_action_names(dataset)
        if not action_names:
            logger.warning("Dataset action names are unavailable. Using uniform frames.")
            return None

        try:
            actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
        except Exception as exc:
            logger.warning("Could not read dataset action column for frame weighting: %s", exc)
            return None
        if actions.ndim != 2:
            logger.warning("Expected action array shape [frames, dim], got %s. Using uniform frames.", actions.shape)
            return None

        name_to_idx = {name: i for i, name in enumerate(action_names)}
        x_idx = name_to_idx.get("x.vel")
        y_idx = name_to_idx.get("y.vel")
        gripper_indices = [
            i for i, name in enumerate(action_names) if "gripper" in name.lower() and name.endswith(".pos")
        ]

        frame_cdfs: list[np.ndarray] = []
        base_hits = 0
        gripper_hits = 0
        placement_hits = 0
        total_valid_frames = 0

        for start, length in zip(self._starts, self._lengths, strict=True):
            end = int(start + length)
            episode_actions = actions[int(start) : end]
            if len(episode_actions) != int(length):
                logger.warning("Action array is shorter than expected. Using uniform frames.")
                return None

            frame_weights = np.ones(int(length), dtype=np.float64)

            if x_idx is not None or y_idx is not None:
                moving = np.zeros(int(length), dtype=bool)
                if x_idx is not None:
                    moving |= np.abs(episode_actions[:, x_idx]) > base_motion_threshold
                if y_idx is not None:
                    moving |= np.abs(episode_actions[:, y_idx]) > base_motion_threshold
                frame_weights[moving] *= base_motion_multiplier
                base_hits += int(moving.sum())

            if gripper_indices and gripper_open_multiplier != 1.0 and int(length) > 1:
                opening = np.zeros(int(length), dtype=bool)
                for idx in gripper_indices:
                    values = episode_actions[:, idx].astype(np.float64)
                    # 自动判断打开方向：默认 episode 末尾更接近“打开/释放”状态。
                    head = values[: max(1, int(0.1 * len(values)))].mean()
                    tail = values[-max(1, int(0.1 * len(values))) :].mean()
                    direction = 1.0 if tail >= head else -1.0
                    delta = np.diff(values, prepend=values[0])
                    opening |= direction * delta > gripper_delta_threshold
                frame_weights[opening] *= gripper_open_multiplier
                gripper_hits += int(opening.sum())

            if placement_multiplier != 1.0 and placement_last_fraction > 0:
                placement_count = int(np.ceil(int(length) * placement_last_fraction))
                if placement_count > 0:
                    placement = np.zeros(int(length), dtype=bool)
                    placement[-placement_count:] = True
                    frame_weights[placement] *= placement_multiplier
                    placement_hits += int(placement.sum())

            frame_weights = np.clip(frame_weights, 1e-12, max_frame_multiplier)
            frame_cdfs.append(np.cumsum(frame_weights) / frame_weights.sum())
            total_valid_frames += int(length)

        logger.info(
            "Frame weighting: frames=%d, base_motion_hits=%d, gripper_open_hits=%d, "
            "placement_hits=%d, multipliers(base=%.2f, gripper=%.2f, placement=%.2f, cap=%.2f)",
            total_valid_frames,
            base_hits,
            gripper_hits,
            placement_hits,
            base_motion_multiplier,
            gripper_open_multiplier,
            placement_multiplier,
            max_frame_multiplier,
        )
        return frame_cdfs

    @staticmethod
    def _get_action_names(dataset: object) -> list[str]:
        features = getattr(getattr(dataset, "meta", None), "features", None)
        if features is None:
            return []
        action_feature = features.get("action") if isinstance(features, dict) else None
        if action_feature is None:
            return []
        if isinstance(action_feature, dict):
            names = action_feature.get("names")
        else:
            names = getattr(action_feature, "names", None)
        return list(names or [])

    @property
    def indices(self) -> list[int]:
        """Materialize one deterministic epoch for introspection."""
        return list(self._iter_epoch(self._epoch, 0))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        generator = self._epoch_generator(epoch)
        # 每个样本先抽 episode，再抽该 episode 内的 frame offset。
        episode_draws = torch.multinomial(
            self._episode_probs,
            num_samples=self._num_samples,
            replacement=True,
            generator=generator,
        ).numpy()
        random_unit = torch.rand(self._num_samples, generator=generator).numpy()
        if self._frame_cdfs is None:
            offsets = (random_unit * self._lengths[episode_draws]).astype(np.int64)
            frame_indices = self._starts[episode_draws] + offsets
        else:
            frame_indices = np.empty(self._num_samples, dtype=np.int64)
            for episode_pos in np.unique(episode_draws):
                sample_mask = episode_draws == episode_pos
                offsets = np.searchsorted(self._frame_cdfs[int(episode_pos)], random_unit[sample_mask], side="right")
                offsets = np.minimum(offsets, self._lengths[int(episode_pos)] - 1)
                frame_indices[sample_mask] = self._starts[int(episode_pos)] + offsets

        for k in range(start, self._num_samples):
            yield int(frame_indices[k])

    def __len__(self) -> int:
        return self._num_samples
