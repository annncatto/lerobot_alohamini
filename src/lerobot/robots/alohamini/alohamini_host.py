#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import time

import cv2
import zmq

from .alohamini import AlohaMini
from .camera_stream import CameraStreamPublisher
from .config_alohamini import AlohaMiniConfig, AlohaMiniHostConfig


class AlohaMiniHost:
    def __init__(self, config: AlohaMiniHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        # Observations are request-driven with a small bounded request window. The Host
        # consumes at most one credit per control loop, preventing unbounded accumulation
        # while allowing transport latency to overlap subsequent observation cycles.
        self.zmq_observation_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_observation_socket.setsockopt(zmq.SNDHWM, config.observation_request_window)
        self.zmq_observation_socket.setsockopt(zmq.RCVHWM, config.observation_request_window)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
 

def _jsonable(value):
    """Convert numpy scalars to JSON-native values without touching normal Python values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def build_observation_multipart(
    observation: dict, camera_keys, encoded_camera_keys=None
) -> list[bytes]:
    """Encode state as JSON and selected camera images as binary JPEG frames."""
    camera_keys = tuple(camera_keys)
    encoded_camera_keys = camera_keys if encoded_camera_keys is None else tuple(encoded_camera_keys)
    state_observation = {
        key: _jsonable(value) for key, value in observation.items() if key not in camera_keys
    }
    state_observation["_image_encoding"] = "jpeg"

    parts = [json.dumps(state_observation).encode("utf-8")]
    image_names = []
    for cam_key in encoded_camera_keys:
        frame = observation.get(cam_key)
        if frame is None:
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            logging.warning("Failed to JPEG encode camera frame %s.", cam_key)
            continue
        image_names.append(cam_key)
        parts.extend([cam_key.encode("utf-8"), buffer.tobytes()])

    state_observation["_images"] = image_names
    parts[0] = json.dumps(state_observation).encode("utf-8")
    return parts


def build_robot_metadata(robot: AlohaMini) -> dict:
    """Describe connected hardware so ROS can validate model and joint units."""
    motors = {}
    for bus in (robot.left_bus, robot.right_bus):
        if bus is None:
            continue
        for name, motor in bus.motors.items():
            calibration = bus.calibration.get(name)
            if calibration is None:
                continue
            motors[name] = {
                "id": int(motor.id),
                "model": motor.model,
                "normalization": motor.norm_mode.value,
                "drive_mode": int(calibration.drive_mode),
                "range_min": int(calibration.range_min),
                "range_max": int(calibration.range_max),
            }

    metadata = {
        "schema_version": 1,
        "robot_model": robot.config.robot_model,
        "motors": motors,
    }
    if getattr(robot, "lift", None) is not None:
        metadata["lift_axis"] = {
            "soft_min_mm": float(robot.lift.cfg.soft_min_mm),
            "soft_max_mm": float(robot.lift.cfg.soft_max_mm),
            "descent_floor_mm": float(robot.lift.cfg.descent_floor_mm),
        }
    return metadata


def observation_request_includes_cameras(request_token: bytes | None) -> bool:
    """Keep legacy/plain requests full while allowing ROS state-only requests."""
    return request_token is not None and not request_token.endswith(b":state")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def main():
    parser = argparse.ArgumentParser(description="Run AlohaMini host process")
    parser.add_argument(
        "--robot_model",
        type=str,
        default="alohamini1",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
        help=(
            "Robot model — drives follower arm profile, base motors, lift motor, lead screw pitch, "
            "and chassis kinematics.\n"
            "  alohamini1   : so-arm-5dof,         base sts3215, lift sts3215, lead=84 mm/rev,  "
            "wheel=0.05m, radius=0.125m\n"
            "  alohamini2   : am-follower-6dof,    base sts3215, lift sts3095, lead=131 mm/rev, "
            "wheel=0.063m, radius=0.195m\n"
            "  alohamini2pro: am-follower-6dof-hd, base sts3250, lift sts3095, lead=131 mm/rev, "
            "wheel=0.063m, radius=0.195m"
        ),
    )
    parser.add_argument(
        "--no_follower",
        action="store_true",
        help="Do not connect follower arms, only operate the base and lift. Use together with --no_leader on the teleoperate side.",
    )
    parser.add_argument(
        "--profile_timing",
        "--profile-timing",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Print average Host, motor, camera, JPEG, and network timings once per second "
            "(default: false)."
        ),
    )
    parser.add_argument(
        "--camera-stream",
        action="store_true",
        help="Publish the optional ROS camera stream on port 5557.",
    )
    parser.add_argument(
        "--camera-stream-port",
        type=int,
        default=None,
        help="Override the dedicated ROS camera PUB port.",
    )
    args = parser.parse_args()

    logging.info("Configuring AlohaMini")
    robot_config = AlohaMiniConfig()
    robot_config.id = "AlohaMiniRobot"
    robot_config.robot_model = args.robot_model
    robot_config.no_follower = args.no_follower
    if args.no_follower:
        logging.info("no_follower mode: follower arms will not connect, only base and lift operate.")
    robot = AlohaMini(robot_config)


    logging.info("Connecting AlohaMini")
    robot.connect()
    robot_metadata = build_robot_metadata(robot)

    logging.info("Starting HostAgent")
    host_config = AlohaMiniHostConfig()
    host_config.camera_stream_enabled = args.camera_stream
    if args.camera_stream_port is not None:
        host_config.port_zmq_camera_stream = args.camera_stream_port
    host = AlohaMiniHost(host_config)
    camera_stream = (
        CameraStreamPublisher(
            robot.cameras,
            port=host_config.port_zmq_camera_stream,
            jpeg_quality=host_config.camera_stream_jpeg_quality,
            max_age_ms=host_config.camera_stream_max_age_ms,
        )
        if host_config.camera_stream_enabled and robot.cameras
        else None
    )
    if camera_stream is not None:
        try:
            camera_stream.start()
        except Exception:
            host.disconnect()
            robot.disconnect()
            raise
        logging.info(
            "Camera stream publishing on tcp://*:%d",
            host_config.port_zmq_camera_stream,
        )

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        timing_report_start_t = start
        timing_loop_count = 0
        timing_totals_ms: dict[str, float] = {}
        timing_command_count = 0
        action_timing_totals_ms: dict[str, float] = {}

        while duration < host.connection_time_s:
            loop_start_t = time.perf_counter()

            # A ``:state`` request deliberately bypasses camera acquisition. Plain
            # legacy request tokens still receive the original full observation.
            request_identity = None
            request_token = None
            try:
                request_parts = host.zmq_observation_socket.recv_multipart(flags=zmq.NOBLOCK)
                request_identity = request_parts[0]
                request_token = request_parts[-1]
            except zmq.Again:
                pass
            request_poll_done_t = time.perf_counter()
            include_cameras = observation_request_includes_cameras(request_token)

            command_received = False
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                #print(f"Received action: {data}")   # debug 
                _action_sent = robot.send_action(data)
                command_received = True
                
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                pass
            except Exception as e:
                logging.exception("Message fetching failed: %s", e)
            command_done_t = time.perf_counter()

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping robot motion."
                )
                watchdog_active = True
                robot.stop_motion()

            # Keep the original Host ordering: apply the newest command before
            # sampling the observation returned in this cycle.
            last_observation = robot.get_observation(include_cameras=include_cameras)
            observation_done_t = time.perf_counter()

            encode_done_t = observation_done_t
            if request_identity is not None and request_token is not None:
                camera_keys = tuple(robot.cameras.keys())
                encoded_camera_keys = () if request_token.endswith(b":state") else camera_keys
                response_observation = {
                    **last_observation,
                    "_robot_metadata": robot_metadata,
                }
                observation_parts = build_observation_multipart(
                    response_observation,
                    camera_keys,
                    encoded_camera_keys,
                )
                encode_done_t = time.perf_counter()
                try:
                    host.zmq_observation_socket.send_multipart(
                        [request_identity, request_token, *observation_parts], flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    logging.info("Dropping observation response, client is not ready")
            response_send_done_t = time.perf_counter()

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = response_send_done_t - loop_start_t

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            loop_done_t = time.perf_counter()

            loop_timings_ms = {
                "command": (command_done_t - request_poll_done_t) * 1e3,
                "robot_observation": (observation_done_t - command_done_t) * 1e3,
                "request_poll": (request_poll_done_t - loop_start_t) * 1e3,
                "jpeg_encode": (encode_done_t - observation_done_t) * 1e3,
                "response_send": (response_send_done_t - encode_done_t) * 1e3,
                "sleep": (loop_done_t - response_send_done_t) * 1e3,
                "loop": (loop_done_t - loop_start_t) * 1e3,
                **robot.logs.get("observation_timing_ms", {}),
            }
            for name, value_ms in loop_timings_ms.items():
                timing_totals_ms[name] = timing_totals_ms.get(name, 0.0) + value_ms
            timing_loop_count += 1
            if command_received:
                for name, value_ms in robot.logs.get("action_timing_ms", {}).items():
                    action_timing_totals_ms[name] = action_timing_totals_ms.get(name, 0.0) + value_ms
                timing_command_count += 1

            timing_elapsed_s = loop_done_t - timing_report_start_t
            if args.profile_timing and timing_elapsed_s >= 1.0:
                averages = {
                    name: total_ms / timing_loop_count for name, total_ms in timing_totals_ms.items()
                }
                camera_text = " ".join(
                    f"{name}={value:.1f}"
                    for name, value in averages.items()
                    if name.startswith("camera_")
                )
                print(
                    f"[HOST TIMING avg ms/loop] Hz={timing_loop_count / timing_elapsed_s:.1f} "
                    f"cmd={averages['command']:.1f} robot_obs={averages['robot_observation']:.1f} "
                    f"left={averages.get('left_arm', 0.0):.1f} base={averages.get('base', 0.0):.1f} "
                    f"right={averages.get('right_arm', 0.0):.1f} lift={averages.get('lift', 0.0):.1f} "
                    f"currents={averages.get('currents', 0.0):.1f} {camera_text} "
                    f"jpeg={averages['jpeg_encode']:.1f} send={averages['response_send']:.1f} "
                    f"sleep={averages['sleep']:.1f} loop={averages['loop']:.1f}",
                    flush=True,
                )
                if camera_stream is not None:
                    camera_stats = camera_stream.stats()
                    print(
                        f"[HOST CAMERA STREAM] published={camera_stats['published']:.0f} "
                        f"dropped={camera_stats['dropped']:.0f} "
                        f"errors={camera_stats['errors']:.0f} "
                        f"encode_avg={camera_stats['average_encode_ms']:.1f}ms",
                        flush=True,
                    )
                if timing_command_count:
                    action_averages = {
                        name: total_ms / timing_command_count
                        for name, total_ms in action_timing_totals_ms.items()
                    }
                    print(
                        f"[HOST ACTION avg ms/command] n={timing_command_count} "
                        f"prepare={action_averages.get('action_prepare', 0.0):.1f} "
                        f"lift={action_averages.get('action_lift', 0.0):.1f} "
                        f"relative={action_averages.get('action_relative_limit', 0.0):.1f} "
                        f"left_gripper_limit={action_averages.get('action_left_gripper_limit', 0.0):.1f} "
                        f"left_joint_limit={action_averages.get('action_left_joint_limit', 0.0):.1f} "
                        f"right_gripper_limit={action_averages.get('action_right_gripper_limit', 0.0):.1f} "
                        f"right_joint_limit={action_averages.get('action_right_joint_limit', 0.0):.1f} "
                        f"left_write={action_averages.get('action_left_write', 0.0):.1f} "
                        f"right_write={action_averages.get('action_right_write', 0.0):.1f} "
                        f"base_write={action_averages.get('action_base_write', 0.0):.1f} "
                        f"total={action_averages.get('action_total', 0.0):.1f}",
                        flush=True,
                    )
                timing_report_start_t = loop_done_t
                timing_loop_count = 0
                timing_totals_ms.clear()
                timing_command_count = 0
                action_timing_totals_ms.clear()

            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down AlohaMini Host.")
        if camera_stream is not None:
            camera_stream.stop()
        robot.disconnect()
        host.disconnect()

    logging.info("Finished AlohaMini cleanly")
if __name__ == "__main__":
    main()
