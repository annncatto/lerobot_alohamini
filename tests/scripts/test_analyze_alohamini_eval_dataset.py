from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_alohamini_eval_dataset import (
    DatasetArrays,
    analyze_dimensions,
    infer_chunk_size,
    integrity_report,
    training_range_tolerance,
)


def test_infer_chunk_size_finds_periodic_replan_jumps(tmp_path):
    rng = np.random.default_rng(7)
    length = 500
    action = np.cumsum(rng.normal(0, 0.02, size=(length, 3)), axis=0)
    action[np.arange(length) % 10 == 0] += rng.normal(
        0,
        2.0,
        size=((np.arange(length) % 10 == 0).sum(), 3),
    )
    frame_index = np.arange(length)
    episode_index = np.zeros(length, dtype=np.int64)

    size, score = infer_chunk_size(action, frame_index, episode_index)

    assert size == 10
    assert score > 1.5


def test_infer_chunk_size_supports_100_step_act_chunks():
    rng = np.random.default_rng(11)
    length = 2400
    action = np.cumsum(rng.normal(0, 0.002, size=(length, 3)), axis=0)
    boundaries = np.arange(100, length, 100)
    action[boundaries] += rng.normal(0, 2.0, size=(len(boundaries), 3))
    frame_index = np.arange(length)
    episode_index = np.zeros(length, dtype=np.int64)

    size, score = infer_chunk_size(action, frame_index, episode_index)

    assert size == 100
    assert score > 1.5


def test_dimension_analysis_separates_boundary_and_inside_jumps(tmp_path):
    length = 200
    action = np.zeros((length, 1))
    for index in range(1, length):
        action[index] = action[index - 1] + (3.0 if index % 10 == 0 else 0.1)
    data = DatasetArrays(
        root=tmp_path,
        info={"fps": 25},
        action=action,
        state=action.copy(),
        timestamp=np.arange(length) / 25,
        frame_index=np.arange(length),
        episode_index=np.zeros(length, dtype=np.int64),
        action_names=["lift_axis.height_mm"],
        state_names=["lift_axis.height_mm"],
    )

    row = analyze_dimensions(data, training=None, chunk_size=10)[0]

    assert row["boundary_jump_ratio"] > 20
    assert row["tracking_mae"] == 0


def test_training_range_tolerance_handles_fixed_dimensions():
    assert training_range_tolerance("lift_axis.height_mm", 230.0, 230.0) == 2.0
    assert training_range_tolerance("x.vel", 0.0, 0.0) == 0.01
    assert training_range_tolerance("arm_right_wrist.pos", 4.0, 4.0) == 0.5


def test_integrity_report_separates_synthetic_and_wall_clock_fps(tmp_path):
    length = 50
    data = DatasetArrays(
        root=tmp_path,
        info={"fps": 25},
        action=np.zeros((length, 1)),
        state=np.zeros((length, 1)),
        timestamp=np.arange(length) / 25,
        frame_index=np.arange(length),
        episode_index=np.zeros(length, dtype=np.int64),
        action_names=["x.vel"],
        state_names=["x.vel"],
    )

    report = integrity_report(data, episode_duration_s=4.0)

    assert report["timestamp_fps"] == pytest.approx(25.0)
    assert report["timestamps_are_synthetic"] is True
    assert report["wall_clock_duration_s"] == 4.0
    assert report["wall_clock_fps"] == 12.5
