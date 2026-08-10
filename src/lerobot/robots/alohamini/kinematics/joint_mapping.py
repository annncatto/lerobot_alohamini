"""Mappings between AlohaMini encoder, LeRobot, URDF and ROS joint spaces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SIDES = ("left", "right")
BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)
ALL_JOINTS = (*BODY_JOINTS, "gripper")
NormalizationMode = Literal["range_m100_100", "range_0_100", "degrees"]


def default_asset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets/alohamini2pro"


def _validate_side(side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")


def signed_tick_delta(tick: int, reference_tick: int, period: int) -> int:
    """Return the shortest signed encoder delta around a periodic register."""
    return int((int(tick) - int(reference_tick) + period // 2) % period - period // 2)


@dataclass(frozen=True)
class JointCalibration:
    name: str
    reference_tick: int
    reference_q_rad: float
    sign: int
    lower_rad: float | None
    upper_rad: float | None
    joint_per_encoder_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError(f"{self.name}: sign must be -1 or +1")
        if self.joint_per_encoder_ratio <= 0:
            raise ValueError(f"{self.name}: joint_per_encoder_ratio must be positive")
        if self.lower_rad is not None and self.upper_rad is not None and self.lower_rad >= self.upper_rad:
            raise ValueError(f"{self.name}: lower limit must be below upper limit")


class AlohaMiniJointMapping:
    """Load the single measured arm calibration and apply it to either side."""

    def __init__(self, asset_dir: str | Path | None = None) -> None:
        self.asset_dir = Path(asset_dir) if asset_dir is not None else default_asset_dir()
        manifest_path = self.asset_dir / "kinematics.json"
        with manifest_path.open(encoding="utf-8") as file:
            self.manifest = json.load(file)
        shared = self.manifest["shared_calibration"]
        if shared["source_side"] != "right":
            raise ValueError("Expected the audited single-arm calibration source")
        if shared["left_alias"].get("inherits") != "hardware_joint_map_right.yaml":
            raise ValueError("Left arm calibration is not pinned to the single measured source")
        self.ticks_per_revolution = int(shared["ticks_per_revolution"])
        self._calibrations = {
            name: self._parse_calibration(name, entry) for name, entry in shared["joints"].items()
        }
        missing = set(ALL_JOINTS) - set(self._calibrations)
        if missing:
            raise ValueError(f"Missing joint calibration entries: {sorted(missing)}")

    def _parse_calibration(self, name: str, entry: dict) -> JointCalibration:
        if name == "gripper":
            lower = float(entry["urdf_open_rad"])
            upper = float(entry["urdf_closed_rad"])
        elif name == "wrist_roll":
            lower, upper = -math.pi, math.pi
        else:
            lower = float(entry["safe_q_min_rad"])
            upper = float(entry["safe_q_max_rad"])
        return JointCalibration(
            name=name,
            reference_tick=int(entry["reference_tick"]),
            reference_q_rad=float(entry["reference_q_rad"]),
            sign=int(entry["sign"]),
            lower_rad=lower,
            upper_rad=upper,
            joint_per_encoder_ratio=float(entry.get("joint_per_encoder_ratio", 1.0)),
        )

    @property
    def joint_order(self) -> tuple[str, ...]:
        return tuple(self.manifest["joint_order"])

    def calibration(self, joint: str) -> JointCalibration:
        try:
            return self._calibrations[joint]
        except KeyError as exc:
            raise KeyError(f"Unknown AlohaMini joint {joint!r}") from exc

    def raw_tick_to_urdf(
        self,
        joint: str,
        tick: int,
        *,
        near_q_rad: float | None = None,
    ) -> float:
        calibration = self.calibration(joint)
        delta = signed_tick_delta(
            tick,
            calibration.reference_tick,
            self.ticks_per_revolution,
        )
        encoder_rad = delta * 2.0 * math.pi / self.ticks_per_revolution
        q_rad = (
            calibration.reference_q_rad + calibration.sign * encoder_rad * calibration.joint_per_encoder_ratio
        )
        # A full-turn wrist has an unavoidable -pi/+pi ambiguity in a 4096-tick
        # register. ROS publishers should pass the previous position so the
        # reported joint remains on the nearest continuous branch.
        if joint == "wrist_roll" and near_q_rad is not None:
            if not math.isfinite(near_q_rad):
                raise ValueError("near_q_rad must be finite")
            period_rad = 2.0 * math.pi * calibration.joint_per_encoder_ratio
            q_rad += round((near_q_rad - q_rad) / period_rad) * period_rad
        return q_rad

    def urdf_to_raw_tick(self, joint: str, q_rad: float, *, check_limits: bool = True) -> int:
        calibration = self.calibration(joint)
        if not math.isfinite(q_rad):
            raise ValueError(f"{joint}: q_rad must be finite")
        if check_limits:
            self.assert_within_limits(joint, q_rad)
        encoder_delta = (
            (q_rad - calibration.reference_q_rad) / calibration.sign / calibration.joint_per_encoder_ratio
        )
        tick_delta = round(encoder_delta * self.ticks_per_revolution / (2.0 * math.pi))
        return int((calibration.reference_tick + tick_delta) % self.ticks_per_revolution)

    def assert_within_limits(self, joint: str, q_rad: float, tolerance: float = 1e-9) -> None:
        calibration = self.calibration(joint)
        if calibration.lower_rad is not None and q_rad < calibration.lower_rad - tolerance:
            raise ValueError(f"{joint}: {q_rad:.6f} rad is below {calibration.lower_rad:.6f} rad")
        if calibration.upper_rad is not None and q_rad > calibration.upper_rad + tolerance:
            raise ValueError(f"{joint}: {q_rad:.6f} rad is above {calibration.upper_rad:.6f} rad")

    def urdf_joint_name(self, side: str, joint: str) -> str:
        _validate_side(side)
        self.calibration(joint)
        suffix = "wrist_yaw_joint" if joint == "wrist_yaw" else joint
        return f"{side}_{suffix}"

    def lerobot_motor_name(self, side: str, joint: str) -> str:
        _validate_side(side)
        self.calibration(joint)
        return f"arm_{side}_{joint}"

    def raw_tick_to_lerobot(
        self,
        tick: int,
        *,
        range_min: int,
        range_max: int,
        drive_mode: int,
        normalization: NormalizationMode,
    ) -> float:
        if range_max <= range_min:
            raise ValueError("range_max must be greater than range_min")
        bounded_tick = min(range_max, max(range_min, int(tick)))
        if normalization == "range_m100_100":
            value = ((bounded_tick - range_min) / (range_max - range_min)) * 200.0 - 100.0
            return -value if drive_mode else value
        if normalization == "range_0_100":
            value = ((bounded_tick - range_min) / (range_max - range_min)) * 100.0
            return 100.0 - value if drive_mode else value
        if normalization == "degrees":
            midpoint = (range_min + range_max) / 2.0
            return (int(tick) - midpoint) * 360.0 / (self.ticks_per_revolution - 1)
        raise ValueError(f"Unknown normalization mode {normalization!r}")

    def lerobot_to_raw_tick(
        self,
        value: float,
        *,
        range_min: int,
        range_max: int,
        drive_mode: int,
        normalization: NormalizationMode,
    ) -> int:
        if range_max <= range_min:
            raise ValueError("range_max must be greater than range_min")
        if normalization == "range_m100_100":
            value = -value if drive_mode else value
            bounded = min(100.0, max(-100.0, value))
            return int(((bounded + 100.0) / 200.0) * (range_max - range_min) + range_min)
        if normalization == "range_0_100":
            value = 100.0 - value if drive_mode else value
            bounded = min(100.0, max(0.0, value))
            return int((bounded / 100.0) * (range_max - range_min) + range_min)
        if normalization == "degrees":
            midpoint = (range_min + range_max) / 2.0
            return int(value * (self.ticks_per_revolution - 1) / 360.0 + midpoint)
        raise ValueError(f"Unknown normalization mode {normalization!r}")

    def lerobot_to_urdf(
        self,
        joint: str,
        value: float,
        *,
        range_min: int,
        range_max: int,
        drive_mode: int,
        normalization: NormalizationMode,
    ) -> float:
        tick = self.lerobot_to_raw_tick(
            value,
            range_min=range_min,
            range_max=range_max,
            drive_mode=drive_mode,
            normalization=normalization,
        )
        return self.raw_tick_to_urdf(joint, tick)

    def urdf_to_lerobot(
        self,
        joint: str,
        q_rad: float,
        *,
        range_min: int,
        range_max: int,
        drive_mode: int,
        normalization: NormalizationMode,
    ) -> float:
        tick = self.urdf_to_raw_tick(joint, q_rad)
        return self.raw_tick_to_lerobot(
            tick,
            range_min=range_min,
            range_max=range_max,
            drive_mode=drive_mode,
            normalization=normalization,
        )
