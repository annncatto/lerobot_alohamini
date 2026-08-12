"""ROS-independent validation and interpolation for joint trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryPointData:
    positions: tuple[float, ...]
    time_s: float


def validate_trajectory(
    joint_names: Sequence[str],
    points: Sequence[TrajectoryPointData],
    *,
    expected_joint_names: Sequence[str],
    position_limits: Mapping[str, tuple[float, float]],
    velocity_limits: Mapping[str, float],
    acceleration_limits: Mapping[str, float],
) -> str | None:
    """Return a rejection reason, or ``None`` when the trajectory is safe."""
    names = tuple(joint_names)
    expected = tuple(expected_joint_names)
    if names != expected:
        return f"joint_names must exactly match {expected}"
    if not points:
        return "trajectory has no points"

    previous_time = -math.inf
    for index, point in enumerate(points):
        if len(point.positions) != len(names):
            return f"point {index} has {len(point.positions)} positions, expected {len(names)}"
        if not math.isfinite(point.time_s) or point.time_s < 0.0 or point.time_s <= previous_time:
            return f"point {index} has a non-increasing or invalid time_from_start"
        previous_time = point.time_s
        for name, position in zip(names, point.positions, strict=True):
            if not math.isfinite(position):
                return f"point {index} joint {name} is not finite"
            lower, upper = position_limits[name]
            if position < lower or position > upper:
                return f"point {index} joint {name}={position:.6f} is outside [{lower:.6f}, {upper:.6f}]"

    segment_velocities: list[tuple[float, ...]] = []
    for index in range(1, len(points)):
        dt = points[index].time_s - points[index - 1].time_s
        velocity = tuple(
            (current - previous) / dt
            for current, previous in zip(points[index].positions, points[index - 1].positions, strict=True)
        )
        for name, value in zip(names, velocity, strict=True):
            if abs(value) > velocity_limits[name] + 1e-9:
                return f"segment {index - 1}->{index} joint {name} velocity {value:.6f} exceeds limit"
        segment_velocities.append(velocity)

    for index in range(1, len(segment_velocities)):
        previous_dt = points[index].time_s - points[index - 1].time_s
        current_dt = points[index + 1].time_s - points[index].time_s
        dt = 0.5 * (previous_dt + current_dt)
        for name, current, previous in zip(
            names, segment_velocities[index], segment_velocities[index - 1], strict=True
        ):
            acceleration = (current - previous) / dt
            if abs(acceleration) > acceleration_limits[name] + 1e-9:
                return f"point {index} joint {name} acceleration {acceleration:.6f} exceeds limit"
    return None


def interpolate_positions(points: Sequence[TrajectoryPointData], elapsed_s: float) -> tuple[float, ...]:
    if not points:
        raise ValueError("trajectory has no points")
    if elapsed_s <= points[0].time_s:
        return points[0].positions
    if elapsed_s >= points[-1].time_s:
        return points[-1].positions
    for previous, current in zip(points, points[1:], strict=False):
        if elapsed_s <= current.time_s:
            ratio = (elapsed_s - previous.time_s) / (current.time_s - previous.time_s)
            return tuple(
                start + ratio * (end - start)
                for start, end in zip(previous.positions, current.positions, strict=True)
            )
    raise RuntimeError("unreachable interpolation state")
