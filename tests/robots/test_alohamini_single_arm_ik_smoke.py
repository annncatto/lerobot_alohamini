from __future__ import annotations

import pytest

from scripts.alohamini_single_arm_ik_smoke_test import build_tick_trajectory, shortest_tick_delta


def test_shortest_tick_delta_wraps_at_encoder_boundary():
    assert shortest_tick_delta(2, 4094) == 4
    assert shortest_tick_delta(4094, 2) == -4


def test_tick_trajectory_limits_each_step_and_reaches_target():
    current = {"a": 4094, "b": 100}
    target = {"a": 2, "b": 107}
    trajectory = build_tick_trajectory(current, target, max_tick_step=2)

    previous = current
    for waypoint in trajectory:
        assert all(abs(shortest_tick_delta(waypoint[name], previous[name])) <= 2 for name in current)
        previous = waypoint
    assert trajectory[-1] == target


def test_tick_trajectory_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="sets differ"):
        build_tick_trajectory({"a": 1}, {"b": 1}, max_tick_step=1)
    with pytest.raises(ValueError, match="positive"):
        build_tick_trajectory({"a": 1}, {"a": 2}, max_tick_step=0)
