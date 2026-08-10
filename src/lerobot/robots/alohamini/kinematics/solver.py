"""AlohaMini2Pro DH forward kinematics and bounded numerical IK."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .joint_mapping import AlohaMiniJointMapping, default_asset_dir


@dataclass(frozen=True)
class IKResult:
    success: bool
    q_rad: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    elapsed_s: float
    min_singular_value: float
    reason: str


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    angle = math.acos(min(1.0, max(-1.0, (trace - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        norm = np.linalg.norm(axis)
        return np.zeros(3) if norm < 1e-12 else axis / norm * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def _pose_error(current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = target[:3, 3] - current[:3, 3]
    orientation = _rotation_vector(target[:3, :3] @ current[:3, :3].T)
    return position, orientation


class AlohaMiniKinematics:
    """Side-parameterized kinematics using the audited shared arm geometry."""

    def __init__(self, side: str, asset_dir: str | Path | None = None) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.asset_dir = Path(asset_dir) if asset_dir is not None else default_asset_dir()
        with (self.asset_dir / "kinematics.json").open(encoding="utf-8") as file:
            self.manifest = json.load(file)
        self.mapping = AlohaMiniJointMapping(self.asset_dir)
        self.joint_order = tuple(self.manifest["joint_order"])
        standard_dh = self.manifest["standard_dh"]
        self.rows = tuple(standard_dh["rows"])
        self.base_transform = np.asarray(standard_dh["base_transform"], dtype=float)
        self.tool_transform = np.asarray(standard_dh["tool_transform"], dtype=float)
        self.tcp_transform = np.eye(4)
        self.tcp_transform[:3, :3] = np.asarray(self.manifest["tcp"]["rotation_fixed_tool"], dtype=float)
        self.tcp_transform[:3, 3] = np.asarray(self.manifest["tcp"]["translation_fixed_m"], dtype=float)
        self.lower_limits = np.asarray(
            [self.mapping.calibration(name).lower_rad for name in self.joint_order],
            dtype=float,
        )
        self.upper_limits = np.asarray(
            [self.mapping.calibration(name).upper_rad for name in self.joint_order],
            dtype=float,
        )
        if not np.all(self.lower_limits < self.upper_limits):
            raise ValueError("IK requires finite ordered limits for every arm joint")

    @staticmethod
    def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array(
            [
                [c, -s * ca, s * sa, a * c],
                [s, c * ca, -c * sa, a * s],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def forward_kinematics(self, q_rad: np.ndarray, *, frame: str = "tcp") -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (len(self.joint_order),):
            raise ValueError(f"Expected q shape {(len(self.joint_order),)}, got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("Joint configuration must be finite")
        transform = self.base_transform.copy()
        for row, value in zip(self.rows, q, strict=True):
            transform = transform @ self._dh_transform(
                float(row["a"]),
                float(row["alpha"]),
                float(row["d"]),
                float(value) + float(row["theta_offset"]),
            )
        fixed_jaw = transform @ self.tool_transform
        if frame == "fixed_jaw":
            return fixed_jaw
        if frame == "tcp":
            return fixed_jaw @ self.tcp_transform
        raise ValueError("frame must be 'fixed_jaw' or 'tcp'")

    def _jacobian(self, q_rad: np.ndarray, current_pose: np.ndarray, epsilon: float) -> np.ndarray:
        jacobian = np.empty((6, len(q_rad)), dtype=float)
        for index in range(len(q_rad)):
            perturbed = q_rad.copy()
            perturbed[index] += epsilon
            pose = self.forward_kinematics(perturbed)
            jacobian[:3, index] = (pose[:3, 3] - current_pose[:3, 3]) / epsilon
            jacobian[3:, index] = _rotation_vector(pose[:3, :3] @ current_pose[:3, :3].T) / epsilon
        return jacobian

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        seed_q_rad: np.ndarray,
        *,
        position_tolerance_m: float = 1e-4,
        orientation_tolerance_rad: float = 1e-3,
        orientation_weight: float = 0.2,
        damping: float = 1e-3,
        max_iterations: int = 200,
        max_joint_step_rad: float = 0.1,
        jacobian_epsilon: float = 1e-6,
    ) -> IKResult:
        started = time.perf_counter()
        target = np.asarray(target_pose, dtype=float)
        seed = np.asarray(seed_q_rad, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            return self._failure(seed, started, "invalid_target")
        if not np.allclose(target[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            return self._failure(seed, started, "invalid_homogeneous_transform")
        rotation = target[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
        ):
            return self._failure(seed, started, "invalid_rotation")
        if seed.shape != (len(self.joint_order),) or not np.all(np.isfinite(seed)):
            return self._failure(seed, started, "invalid_seed")
        if np.any(seed < self.lower_limits) or np.any(seed > self.upper_limits):
            return self._failure(seed, started, "seed_out_of_limits")
        if orientation_weight < 0 or position_tolerance_m <= 0 or orientation_tolerance_rad <= 0:
            raise ValueError("IK weights and tolerances must be valid")
        if damping <= 0 or max_iterations < 1 or max_joint_step_rad <= 0:
            raise ValueError("IK damping, iterations and step limit must be positive")

        q = seed.copy()
        min_singular_value = 0.0
        for iteration in range(max_iterations + 1):
            current = self.forward_kinematics(q)
            position_error, orientation_error = _pose_error(current, target)
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(orientation_error))
            orientation_ok = orientation_weight == 0 or orientation_norm <= orientation_tolerance_rad
            if position_norm <= position_tolerance_m and orientation_ok:
                return IKResult(
                    success=True,
                    q_rad=q.copy(),
                    position_error_m=position_norm,
                    orientation_error_rad=orientation_norm,
                    iterations=iteration,
                    elapsed_s=time.perf_counter() - started,
                    min_singular_value=min_singular_value,
                    reason="converged",
                )
            if iteration == max_iterations:
                break

            jacobian = self._jacobian(q, current, jacobian_epsilon)
            weights = np.array([1.0, 1.0, 1.0, orientation_weight, orientation_weight, orientation_weight])
            weighted_jacobian = weights[:, None] * jacobian
            weighted_error = weights * np.concatenate([position_error, orientation_error])
            singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
            min_singular_value = float(singular_values[-1])
            system = weighted_jacobian @ weighted_jacobian.T + damping**2 * np.eye(6)
            try:
                delta = weighted_jacobian.T @ np.linalg.solve(system, weighted_error)
            except np.linalg.LinAlgError:
                return self._result_from_pose(
                    False, q, target, iteration, started, min_singular_value, "linear_solve_failed"
                )
            delta = np.clip(delta, -max_joint_step_rad, max_joint_step_rad)
            q = np.clip(q + delta, self.lower_limits, self.upper_limits)

        return self._result_from_pose(
            False,
            q,
            target,
            max_iterations,
            started,
            min_singular_value,
            "max_iterations",
        )

    def _failure(self, q: np.ndarray, started: float, reason: str) -> IKResult:
        safe_q = q.copy() if q.ndim == 1 else np.empty(0)
        return IKResult(
            success=False,
            q_rad=safe_q,
            position_error_m=math.inf,
            orientation_error_rad=math.inf,
            iterations=0,
            elapsed_s=time.perf_counter() - started,
            min_singular_value=0.0,
            reason=reason,
        )

    def _result_from_pose(
        self,
        success: bool,
        q: np.ndarray,
        target: np.ndarray,
        iterations: int,
        started: float,
        min_singular_value: float,
        reason: str,
    ) -> IKResult:
        current = self.forward_kinematics(q)
        position_error, orientation_error = _pose_error(current, target)
        return IKResult(
            success=success,
            q_rad=q.copy(),
            position_error_m=float(np.linalg.norm(position_error)),
            orientation_error_rad=float(np.linalg.norm(orientation_error)),
            iterations=iterations,
            elapsed_s=time.perf_counter() - started,
            min_singular_value=min_singular_value,
            reason=reason,
        )
