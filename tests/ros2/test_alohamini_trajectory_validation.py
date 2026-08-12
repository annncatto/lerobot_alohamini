from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "ros2/alohamini_control"
sys.path.insert(0, str(PACKAGE_ROOT))

from alohamini_control.trajectory_validation import (  # noqa: E402
    TrajectoryPointData,
    interpolate_positions,
    validate_trajectory,
)

NAMES = ("joint_a", "joint_b")
POSITION_LIMITS = dict.fromkeys(NAMES, (-1.0, 1.0))
VELOCITY_LIMITS = dict.fromkeys(NAMES, 1.0)
ACCELERATION_LIMITS = dict.fromkeys(NAMES, 2.0)


def validate(points, names=NAMES):
    return validate_trajectory(
        names,
        points,
        expected_joint_names=NAMES,
        position_limits=POSITION_LIMITS,
        velocity_limits=VELOCITY_LIMITS,
        acceleration_limits=ACCELERATION_LIMITS,
    )


def test_validates_and_interpolates_safe_trajectory():
    points = (
        TrajectoryPointData((0.0, 0.0), 0.0),
        TrajectoryPointData((0.5, -0.5), 1.0),
    )
    assert validate(points) is None
    assert interpolate_positions(points, 0.5) == pytest.approx((0.25, -0.25))


@pytest.mark.parametrize(
    ("points", "names", "reason"),
    [
        ((), NAMES, "no points"),
        ((TrajectoryPointData((0.0, 0.0), 0.0),), tuple(reversed(NAMES)), "exactly match"),
        ((TrajectoryPointData((2.0, 0.0), 0.0),), NAMES, "outside"),
        (
            (TrajectoryPointData((0.0, 0.0), 0.0), TrajectoryPointData((0.0, 0.0), 0.0)),
            NAMES,
            "non-increasing",
        ),
        (
            (TrajectoryPointData((0.0, 0.0), 0.0), TrajectoryPointData((0.5, 0.0), 0.1)),
            NAMES,
            "velocity",
        ),
    ],
)
def test_rejects_unsafe_trajectory(points, names, reason):
    assert reason in validate(points, names)
