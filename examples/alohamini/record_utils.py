#!/usr/bin/env python3

import logging
import time
from collections.abc import Callable
from typing import Any

from lerobot.datasets import LeRobotDataset, safe_stop_image_writer
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_visualization_data


@safe_stop_image_writer
def record_loop(
    robot: Any,
    events: dict,
    fps: int,
    leader_arm: Any,
    keyboard: Any,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    timing_callback: Callable[[dict[str, float]], None] | None = None,
    display_data: bool = False,
    control_fps: int = 50,
) -> None:
    """Run 50 Hz control while committing timestamp-aligned camera frames at ``fps``."""
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    if control_time_s is None:
        raise ValueError("control_time_s must be provided")

    if control_fps < fps:
        raise ValueError(f"control_fps must be at least dataset fps ({control_fps} < {fps}).")

    control_interval = 1 / control_fps
    camera_interval = 1 / fps
    start_episode_t = time.perf_counter()
    next_camera_request_t = start_episode_t
    timestamp = 0.0
    control_report_count = 0
    dataset_report_count = 0
    fps_report_start_t = start_episode_t
    last_recorded_camera_timestamps: dict[str, float] = {}
    previous_control_sample: dict[str, Any] | None = None
    # Estimate PC-monotonic minus Host-monotonic from the lowest observed
    # receive latency. This lets leader actions be compared with Host camera
    # timestamps without requiring synchronized wall clocks.
    host_clock_offset_estimate: float | None = None
    timing_totals_s = {
        "observation": 0.0,
        "observation_processing": 0.0,
        "frame_build": 0.0,
        "teleop": 0.0,
        "send_action": 0.0,
        "dataset_write": 0.0,
        "display": 0.0,
        "sleep": 0.0,
        "loop": 0.0,
    }

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Request images at dataset cadence, but request state and run teleoperation
        # every control tick. Host camera capture is always asynchronous and its
        # 50 Hz control thread only peeks the latest frame cache.
        request_cameras = start_loop_t >= next_camera_request_t
        if request_cameras:
            while next_camera_request_t <= start_loop_t:
                next_camera_request_t += camera_interval
        obs = robot.get_observation(include_cameras=request_cameras)
        observation_done_t = time.perf_counter()
        for name, value_ms in getattr(robot, "logs", {}).get("observation_timing_ms", {}).items():
            timing_totals_s[name] = timing_totals_s.get(name, 0.0) + value_ms / 1000

        obs_processed = robot_observation_processor(obs)
        observation_processing_done_t = time.perf_counter()

        frame_build_done_t = time.perf_counter()

        arm_action = {f"arm_{key}": value for key, value in leader_arm.get_action().items()}
        keyboard_action = keyboard.get_action()
        action = {
            **arm_action,
            **robot._from_keyboard_to_base_action(keyboard_action),
            **robot._from_keyboard_to_lift_action(keyboard_action),
        }
        action_values = teleop_action_processor((action, obs))
        robot_action_to_send = robot_action_processor((action_values, obs))
        teleop_done_t = time.perf_counter()

        robot.send_action(robot_action_to_send)
        send_action_done_t = time.perf_counter()

        host_timing = dict(getattr(robot, "latest_host_timing", {}))
        state_started_t = host_timing.get("state_sample_started_monotonic_s")
        state_finished_t = host_timing.get("state_sample_finished_monotonic_s")
        if state_finished_t is not None:
            offset_sample = observation_done_t - float(state_finished_t)
            host_clock_offset_estimate = (
                offset_sample
                if host_clock_offset_estimate is None
                else min(host_clock_offset_estimate, offset_sample)
            )
        state_timestamp = (
            (float(state_started_t) + float(state_finished_t)) / 2
            if state_started_t is not None and state_finished_t is not None
            else None
        )
        current_sample = {
            "observation": obs_processed,
            "action": action_values,
            "state_timestamp": state_timestamp,
            "action_timestamp": (
                teleop_done_t - host_clock_offset_estimate
                if host_clock_offset_estimate is not None
                else None
            ),
        }

        camera_timestamps = {
            str(name): float(value)
            for name, value in host_timing.get("camera_capture_monotonic_s", {}).items()
        }
        expected_cameras = set(getattr(robot, "_cameras_ft", {}))
        camera_sample_ready = bool(expected_cameras) and all(
            name in camera_timestamps
            and camera_timestamps[name] != last_recorded_camera_timestamps.get(name)
            for name in expected_cameras
        )

        if dataset is not None and camera_sample_ready:
            camera_timestamp = sum(camera_timestamps[name] for name in expected_cameras) / len(
                expected_cameras
            )
            candidates = [current_sample]
            if previous_control_sample is not None:
                candidates.append(previous_control_sample)
            timestamped_candidates = [
                sample for sample in candidates if sample["state_timestamp"] is not None
            ]
            aligned_state_sample = (
                min(
                    timestamped_candidates,
                    key=lambda sample: abs(sample["state_timestamp"] - camera_timestamp),
                )
                if timestamped_candidates
                else current_sample
            )
            action_candidates = [
                sample for sample in candidates if sample["action_timestamp"] is not None
            ]
            aligned_action_sample = (
                min(
                    action_candidates,
                    key=lambda sample: abs(sample["action_timestamp"] - camera_timestamp),
                )
                if action_candidates
                else current_sample
            )
            aligned_observation = dict(aligned_state_sample["observation"])
            # State and desired action are independently chosen from adjacent
            # 50 Hz samples. Images always come from the newly observed camera
            # timestamp set.
            for camera_name in expected_cameras:
                aligned_observation[camera_name] = obs_processed[camera_name]
            observation_frame = build_dataset_frame(
                dataset.features, aligned_observation, prefix=OBS_STR
            )
            action_frame = build_dataset_frame(
                dataset.features, aligned_action_sample["action"], prefix=ACTION
            )
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
            last_recorded_camera_timestamps = camera_timestamps
            dataset_report_count += 1
        dataset_write_done_t = time.perf_counter()

        if display_data:
            log_visualization_data(
                "rerun",
                observation=obs_processed,
                action=action_values,
            )
        work_done_t = time.perf_counter()
        work_duration_s = work_done_t - start_loop_t
        sleep_time_s = control_interval - work_duration_s
        if sleep_time_s < 0:
            logging.warning(
                "AlohaMini record loop is running slower (%.1f Hz) than the target FPS (%d Hz).",
                1 / work_duration_s,
                control_fps,
            )
        precise_sleep(max(sleep_time_s, 0.0))
        loop_done_t = time.perf_counter()

        timing_totals_s["observation"] += observation_done_t - start_loop_t
        timing_totals_s["observation_processing"] += (
            observation_processing_done_t - observation_done_t
        )
        timing_totals_s["frame_build"] += frame_build_done_t - observation_processing_done_t
        timing_totals_s["teleop"] += teleop_done_t - frame_build_done_t
        timing_totals_s["send_action"] += send_action_done_t - teleop_done_t
        timing_totals_s["dataset_write"] += dataset_write_done_t - send_action_done_t
        timing_totals_s["display"] += work_done_t - dataset_write_done_t
        timing_totals_s["sleep"] += loop_done_t - work_done_t
        timing_totals_s["loop"] += loop_done_t - start_loop_t

        previous_control_sample = current_sample
        control_report_count += 1
        report_now_t = time.perf_counter()
        fps_report_elapsed_s = report_now_t - fps_report_start_t
        if fps_report_elapsed_s >= 1.0:
            control_rate = control_report_count / fps_report_elapsed_s
            dataset_rate = dataset_report_count / fps_report_elapsed_s
            if timing_callback is not None:
                timing_callback(
                    {
                        "capture_fps": dataset_rate,
                        "control_fps": control_rate,
                        **{
                            name: total_s * 1000 / control_report_count
                            for name, total_s in timing_totals_s.items()
                        },
                    }
                )
            for name in timing_totals_s:
                timing_totals_s[name] = 0.0
            control_report_count = 0
            dataset_report_count = 0
            fps_report_start_t = report_now_t

        timestamp = time.perf_counter() - start_episode_t
