#!/usr/bin/env python3

from collections.abc import Mapping


def snap_planar_velocity(
    action: Mapping[str, float],
    *,
    speed: float = 0.15,
    deadband: float = 0.05,
) -> dict[str, float]:
    """Snap AlohaMini planar velocity predictions to ``{-speed, 0, +speed}``.

    Only ``x.vel`` and ``y.vel`` are changed.  A non-zero deadband is important:
    ACT's uncertain prediction is often close to the dataset mean (about 0.02
    for the current x velocity), and mapping every positive number to +speed
    would turn numerical noise into full-speed motion.
    """
    if speed < 0:
        raise ValueError("speed must be non-negative")
    if deadband < 0:
        raise ValueError("deadband must be non-negative")

    snapped = dict(action)
    if speed == 0:
        return snapped

    for key in ("x.vel", "y.vel"):
        if key not in snapped:
            continue
        value = float(snapped[key])
        if value > deadband:
            snapped[key] = speed
        elif value < -deadband:
            snapped[key] = -speed
        else:
            snapped[key] = 0.0
    return snapped
