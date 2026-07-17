#!/usr/bin/env python3

import argparse
import json
import shlex
import socket
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import torch

import lerobot.robots.alohamini  # noqa: F401 - registers alohamini robot configs
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.robot_client import RobotClient
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini import AlohaMiniClientConfig, AlohaMiniConfig
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.action_quantization import snap_planar_velocity
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def log_eval(message: str) -> None:
    print(message, flush=True)
    log_say(message, play_sounds=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AlohaMini with a policy running on a remote LeRobot policy server"
    )

    # Keep the evaluation, dataset, policy, and robot flags aligned with evaluate_bi.py.
    parser.add_argument("--eval.n_episodes", "--num_episodes", dest="num_episodes", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--eval.recording_fps",
        dest="recording_fps",
        type=int,
        default=5,
        help="Video/dataset recording rate. Action commands still run at --fps.",
    )
    parser.add_argument(
        "--eval.record_dataset",
        dest="record_dataset",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Record observations and actions as a LeRobotDataset during evaluation.",
    )
    parser.add_argument("--eval.episode_time_s", "--episode_time", dest="episode_time", type=int, default=60)
    parser.add_argument(
        "--dataset.single_task",
        "--task_description",
        dest="task_description",
        type=str,
        default="robot task",
    )
    parser.add_argument("--policy.path", "--hf_model_id", dest="policy_path", type=str, required=True)
    parser.add_argument(
        "--policy.type",
        "--policy_type",
        dest="policy_type",
        choices=SUPPORTED_POLICIES,
        help="Policy type. If omitted, read it from the checkpoint config.json locally or over SSH.",
    )
    parser.add_argument("--dataset.repo_id", "--hf_dataset_id", dest="dataset_repo_id", type=str)
    parser.add_argument(
        "--dataset.push_to_hub",
        "--push_to_hub",
        dest="push_to_hub",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--robot.remote_ip", "--remote_ip", dest="robot_ip", default="127.0.0.1")
    parser.add_argument("--robot.id", "--robot_id", dest="robot_id", default="my_alohamini")
    parser.add_argument(
        "--robot.transport",
        dest="robot_transport",
        choices=["zmq", "direct"],
        default="zmq",
        help=(
            "zmq: connect to an AlohaMini Host process (the historical PC/Host path). "
            "direct: open the robot motors and cameras in this process; use this when the evaluator "
            "runs on the Raspberry Pi and connects straight to the remote policy server."
        ),
    )
    parser.add_argument(
        "--robot.left_port",
        dest="robot_left_port",
        default="/dev/am_arm_follower_left",
        help="Follower-left serial port used by --robot.transport=direct.",
    )
    parser.add_argument(
        "--robot.right_port",
        dest="robot_right_port",
        default="/dev/am_arm_follower_right",
        help="Follower-right serial port used by --robot.transport=direct.",
    )
    parser.add_argument(
        "--robot.robot_model",
        "--robot_model",
        dest="robot_model",
        default="alohamini1",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
    )

    parser.add_argument(
        "--inference.type",
        "--inference_type",
        dest="inference_type",
        default="async",
        choices=["async", "sync", "rtc"],
        help="async: ordinary remote action chunks (required for policies without RTC, such as "
        "VLA-JEPA). sync: one remote select_action call at a time. rtc: remote RTC prefix-inpainting.",
    )
    parser.add_argument(
        "--inference.rtc.execution_horizon",
        dest="rtc_execution_horizon",
        type=int,
        default=10,
        help="RTC only: number of prefix steps re-generated per inference call.",
    )
    parser.add_argument(
        "--inference.rtc.max_guidance_weight",
        dest="rtc_max_guidance_weight",
        type=float,
        default=10.0,
        help="RTC only: maximum prefix-inpainting guidance weight.",
    )
    parser.add_argument(
        "--inference.rtc.queue_threshold",
        dest="rtc_queue_threshold",
        type=int,
        default=30,
        help="RTC only: request another chunk when at most this many actions remain.",
    )
    parser.add_argument(
        "--interpolation_multiplier",
        dest="interpolation_multiplier",
        type=int,
        default=1,
        help="Send N interpolated commands per policy action (1 disables interpolation).",
    )
    parser.add_argument(
        "--action.base_snap_speed",
        dest="base_snap_speed",
        type=float,
        default=0.15,
        help="Snap x.vel/y.vel to {-speed, 0, +speed}; set 0 to disable.",
    )
    parser.add_argument(
        "--action.base_snap_deadband",
        dest="base_snap_deadband",
        type=float,
        default=0.05,
        help="Predicted |x.vel|/|y.vel| at or below this value is sent as zero.",
    )

    parser.add_argument("--async.actions_per_chunk", dest="actions_per_chunk", type=int, default=50)
    parser.add_argument("--async.chunk_size_threshold", dest="chunk_size_threshold", type=float, default=0.6)
    parser.add_argument(
        "--async.aggregate_fn",
        dest="aggregate_fn_name",
        default="weighted_average",
        choices=["weighted_average", "latest_only", "average", "conservative"],
    )
    parser.add_argument("--async.client_device", dest="client_device", default="cpu")
    parser.add_argument(
        "--async.image_compression_quality",
        "--observation.jpeg_quality",
        dest="image_compression_quality",
        type=int,
        default=85,
        help=(
            "JPEG quality used before sending camera observations. Set to 0 to send raw image arrays; "
            "raw transport is normally too large for three 640x480 cameras over a WAN."
        ),
    )
    parser.add_argument(
        "--observation.send_mode",
        dest="observation_send_mode",
        choices=["latest", "queue_gated"],
        default="latest",
        help=(
            "latest: submit the newest observation every --fps tick for every policy type, replacing "
            "any pending stale frame. queue_gated: preserve the original async action-queue trigger."
        ),
    )
    parser.add_argument("--policy.device", dest="policy_device", default="cuda")
    parser.add_argument(
        "--policy.rename_map",
        dest="policy_rename_map",
        type=json.loads,
        default={},
        help="JSON mapping from robot observation feature names to policy feature names.",
    )

    # The script can either manage an SSH-hosted server or connect to an already reachable server.
    parser.add_argument("--server.address", dest="server_address")
    parser.add_argument("--remote.manage_server", dest="manage_server", type=parse_bool, default=True)
    parser.add_argument("--remote.host", dest="remote_host", default="connect.bjb1.seetacloud.com")
    parser.add_argument("--remote.user", dest="remote_user", default="root")
    parser.add_argument("--remote.ssh_port", dest="remote_ssh_port", type=int, default=22)
    parser.add_argument("--remote.identity_file", dest="remote_identity_file")
    parser.add_argument(
        "--remote.repo",
        dest="remote_repo",
        default="/root/autodl-tmp/pi_train/repos/lerobot_alohamini",
    )
    parser.add_argument(
        "--remote.python",
        dest="remote_python",
        default="/root/autodl-tmp/pi_train/envs/lerobot/bin/python",
    )
    parser.add_argument(
        "--remote.hf_home", dest="remote_hf_home", default="/root/autodl-tmp/pi_train/hf_cache"
    )
    parser.add_argument(
        "--remote.log",
        dest="remote_log",
        default="/root/autodl-tmp/pi_train/logs/policy_server.log",
    )
    parser.add_argument("--remote.server_port", dest="remote_server_port", type=int, default=8080)
    parser.add_argument("--remote.local_port", dest="local_tunnel_port", type=int, default=18080)
    parser.add_argument("--remote.offline", dest="remote_offline", type=parse_bool, default=True)
    parser.add_argument("--remote.connect_timeout_s", dest="connect_timeout_s", type=float, default=20.0)
    return parser


def ssh_base(args: argparse.Namespace) -> list[str]:
    command = [
        "ssh",
        "-p",
        str(args.remote_ssh_port),
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if args.remote_identity_file:
        command.extend(["-i", args.remote_identity_file])
    command.append(f"{args.remote_user}@{args.remote_host}")
    return command


def run_remote(args: argparse.Namespace, script: str) -> subprocess.CompletedProcess[str]:
    remote_command = f"bash -lc {shlex.quote(script)}"
    try:
        return subprocess.run(
            [*ssh_base(args), remote_command],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or f"exit status {error.returncode}"
        raise RuntimeError(f"Remote SSH command failed: {detail}") from error


def infer_policy_type(args: argparse.Namespace) -> str:
    if args.policy_type:
        return args.policy_type

    local_config = Path(args.policy_path) / "config.json"
    if local_config.is_file():
        policy_type = json.loads(local_config.read_text())["type"]
    elif args.manage_server:
        config_path = shlex.quote(str(Path(args.policy_path) / "config.json"))
        result = run_remote(
            args,
            f"test -f {config_path} || "
            f"{{ echo 'Remote checkpoint config not found: {config_path}' >&2; exit 2; }}; "
            f"{shlex.quote(args.remote_python)} -c "
            + shlex.quote("import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['type'])")
            + f" {config_path}",
        )
        policy_type = result.stdout.strip().splitlines()[-1]
    else:
        raise ValueError("--policy.type is required when the checkpoint config is not available locally")

    if policy_type not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Policy type {policy_type!r} is not supported by async inference; "
            f"choose one of {SUPPORTED_POLICIES}"
        )
    return policy_type


def start_remote_server(args: argparse.Namespace) -> None:
    offline = "1" if args.remote_offline else "0"
    pid_file = args.remote_log + ".pid"
    server_command = shlex.join(
        [
            "env",
            f"PYTHONPATH={args.remote_repo}/src",
            f"HF_HOME={args.remote_hf_home}",
            f"HF_HUB_OFFLINE={offline}",
            f"TRANSFORMERS_OFFLINE={offline}",
            args.remote_python,
            "-u",
            "-m",
            "lerobot.async_inference.policy_server",
            "--host=127.0.0.1",
            f"--port={args.remote_server_port}",
            f"--fps={args.fps}",
            "--inference_latency=0",
            "--obs_queue_timeout=10",
        ]
    )
    script = "\n".join(
        [
            "set -e",
            f"mkdir -p {shlex.quote(str(Path(args.remote_log).parent))}",
            f"if test -s {shlex.quote(pid_file)} && "
            f'kill -0 "$(cat {shlex.quote(pid_file)})" 2>/dev/null; then',
            "  echo 'Remote policy server already running'",
            "else",
            f"  cd {shlex.quote(args.remote_repo)}",
            f"  nohup {server_command} > {shlex.quote(args.remote_log)} 2>&1 < /dev/null &",
            f"  echo $! > {shlex.quote(pid_file)}",
            '  echo "Started remote policy server pid=$!"',
            "fi",
            "server_ready=",
            "for _ in {1..80}; do",
            f"  if (: > /dev/tcp/127.0.0.1/{args.remote_server_port}) 2>/dev/null; then",
            "    server_ready=1",
            "    break",
            "  fi",
            "  sleep 0.25",
            "done",
            'if test "$server_ready" != 1; then',
            "  echo 'Remote policy server did not become ready' >&2",
            f"  tail -n 80 {shlex.quote(args.remote_log)} >&2 || true",
            "  exit 3",
            "fi",
        ]
    )
    result = run_remote(args, script)
    if result.stdout.strip():
        print(result.stdout.strip())


def start_ssh_tunnel(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    command = ssh_base(args)
    command[-1:-1] = [
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"{args.local_tunnel_port}:127.0.0.1:{args.remote_server_port}",
    ]
    return subprocess.Popen(command)


def wait_for_server(address: str, timeout_s: float, tunnel: subprocess.Popen[bytes] | None) -> None:
    host, port_text = address.rsplit(":", 1)
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        if tunnel is not None and tunnel.poll() is not None:
            raise RuntimeError(f"SSH tunnel exited with status {tunnel.returncode}")
        try:
            with socket.create_connection((host, int(port_text)), timeout=1):
                return
        except OSError as error:
            last_error = error
            time.sleep(0.25)
    raise TimeoutError(f"Policy server at {address} was not ready after {timeout_s}s: {last_error}")


def stop_tunnel(tunnel: subprocess.Popen[bytes] | None) -> None:
    if tunnel is None or tunnel.poll() is not None:
        return
    tunnel.terminate()
    try:
        tunnel.wait(timeout=5)
    except subprocess.TimeoutExpired:
        tunnel.kill()
        tunnel.wait(timeout=5)


def pop_action(client: RobotClient) -> tuple[dict[str, float], int]:
    with client.action_queue_lock:
        client.action_queue_size.append(client.action_queue.qsize())
        timed_action = client.action_queue.get_nowait()
    action = {
        key: timed_action.get_action()[index].item() for index, key in enumerate(client.robot.action_features)
    }
    with client.latest_action_lock:
        client.latest_action = timed_action.get_timestep()
    return action, timed_action.get_timestep()


def send_base_stop(client: RobotClient) -> None:
    """Stop mobile-base motion without changing arm or lift position targets."""
    client.robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})


def run_action_control_loop(
    client: RobotClient,
    fps: int,
    interpolation_multiplier: int,
    stop_event: threading.Event,
    state: dict[str, Any],
    state_lock: threading.Lock,
    base_snap_speed: float,
    base_snap_deadband: float,
) -> None:
    """Execute queued actions at a stable rate, independently of camera recording."""
    interpolator = ActionInterpolator(multiplier=interpolation_multiplier)
    interval = interpolator.get_control_interval(fps)
    next_tick = time.perf_counter()
    base_stopped = False

    try:
        while not stop_event.is_set():
            if interpolator.needs_new_action() and client.actions_available():
                try:
                    queued_action, _ = pop_action(client)
                    interpolator.add(
                        torch.tensor([queued_action[key] for key in client.robot.action_features])
                    )
                except Empty:
                    # The receive thread may replace the queue between the availability check and pop.
                    pass

            action_tensor = interpolator.get()
            action = None
            if action_tensor is not None:
                action = {
                    key: action_tensor[index].item() for index, key in enumerate(client.robot.action_features)
                }

            if action is not None:
                action = snap_planar_velocity(
                    action,
                    speed=base_snap_speed,
                    deadband=base_snap_deadband,
                )
                client.robot.send_action(action)
                with state_lock:
                    state["latest_action"] = action
                    state["actions_executed"] += 1
                base_stopped = False
            elif not base_stopped:
                send_base_stop(client)
                with state_lock:
                    latest_action = state.get("latest_action")
                    if latest_action is not None:
                        state["latest_action"] = {
                            **latest_action,
                            "x.vel": 0.0,
                            "y.vel": 0.0,
                            "theta.vel": 0.0,
                        }
                    state["queue_underruns"] += 1
                base_stopped = True

            next_tick += interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time <= 0:
                with state_lock:
                    state["control_overruns"] += 1
                next_tick = time.perf_counter()
                continue
            stop_event.wait(sleep_time)
    except Exception as error:
        with state_lock:
            state["error"] = error
        stop_event.set()
    finally:
        with suppress(Exception):
            send_base_stop(client)


@dataclass
class PendingObservation:
    raw: dict[str, Any]
    received_timestamp: float


class LatestObservationSender:
    """Upload observations without blocking capture; pending backlog is always latest-only."""

    def __init__(self, client: RobotClient, task: str):
        self.client = client
        self.task = task
        self.queue: Queue[PendingObservation] = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="observation-uploader", daemon=True)
        self.lock = threading.Lock()
        self.submitted = 0
        self.replaced = 0
        self.sent = 0
        self.queue_delay_seconds = 0.0
        self.max_queue_delay_seconds = 0.0
        self.error: Exception | None = None

    def start(self) -> None:
        self.thread.start()

    def submit(self, raw: dict[str, Any], received_timestamp: float) -> None:
        pending = PendingObservation(raw=raw, received_timestamp=received_timestamp)
        with self.lock:
            self.submitted += 1
        try:
            self.queue.put_nowait(pending)
            return
        except Full:
            pass

        # The upload worker is still sending an older observation. Replace only the pending item;
        # never build an unbounded queue of stale camera frames.
        try:
            _ = self.queue.get_nowait()
        except Empty:
            pass
        else:
            with self.lock:
                self.replaced += 1
        try:
            self.queue.put_nowait(pending)
        except Full:
            # The worker raced us and another producer cannot exist, so dropping here is harmless:
            # either this item or a frame captured within the same instant is already pending.
            with self.lock:
                self.replaced += 1

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    pending = self.queue.get(timeout=0.05)
                except Empty:
                    continue
                delay = max(0.0, time.time() - pending.received_timestamp)
                self.client.control_loop_observation(
                    self.task,
                    verbose=False,
                    raw_observation=pending.raw,
                    observation_timestamp=pending.received_timestamp,
                )
                with self.lock:
                    self.sent += 1
                    self.queue_delay_seconds += delay
                    self.max_queue_delay_seconds = max(self.max_queue_delay_seconds, delay)
        except Exception as error:
            with self.lock:
                self.error = error
            self.stop_event.set()

    def snapshot(self) -> dict[str, float | int | Exception | None]:
        with self.lock:
            mean_delay = self.queue_delay_seconds / self.sent if self.sent else 0.0
            return {
                "submitted": self.submitted,
                "replaced": self.replaced,
                "sent": self.sent,
                "mean_queue_delay_ms": mean_delay * 1000,
                "max_queue_delay_ms": self.max_queue_delay_seconds * 1000,
                "error": self.error,
            }

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.recording_fps <= 0:
        raise ValueError("--eval.recording_fps must be positive")
    if args.recording_fps > args.fps:
        raise ValueError("--eval.recording_fps cannot exceed --fps")
    if args.record_dataset and not args.dataset_repo_id:
        raise ValueError("--dataset.repo_id is required when --eval.record_dataset=true")
    if args.num_episodes <= 0:
        raise ValueError("--eval.n_episodes must be positive")
    if args.episode_time <= 0:
        raise ValueError("--eval.episode_time_s must be positive")
    if args.rtc_execution_horizon <= 0:
        raise ValueError("--inference.rtc.execution_horizon must be positive")
    if args.rtc_max_guidance_weight <= 0:
        raise ValueError("--inference.rtc.max_guidance_weight must be positive")
    if args.rtc_queue_threshold < 0:
        raise ValueError("--inference.rtc.queue_threshold cannot be negative")
    if args.interpolation_multiplier <= 0:
        raise ValueError("--interpolation_multiplier must be positive")
    if args.base_snap_speed < 0:
        raise ValueError("--action.base_snap_speed cannot be negative")
    if args.base_snap_deadband < 0:
        raise ValueError("--action.base_snap_deadband cannot be negative")
    if args.inference_type == "rtc" and args.rtc_queue_threshold >= args.actions_per_chunk:
        raise ValueError("RTC queue_threshold must be smaller than --async.actions_per_chunk")
    if not isinstance(args.policy_rename_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in args.policy_rename_map.items()
    ):
        raise ValueError("--policy.rename_map must be a JSON object mapping strings to strings")
    if args.manage_server and args.server_address:
        raise ValueError("Use either --remote.manage_server=true or --server.address, not both")
    if not args.manage_server and not args.server_address:
        raise ValueError("--server.address is required when --remote.manage_server=false")

    tunnel: subprocess.Popen[bytes] | None = None
    client: RobotClient | None = None
    dataset: LeRobotDataset | None = None
    action_receiver_thread: threading.Thread | None = None
    action_control_thread: threading.Thread | None = None
    observation_sender: LatestObservationSender | None = None
    action_control_stop = threading.Event()
    action_state_lock = threading.Lock()
    action_state: dict[str, Any] = {
        "latest_action": None,
        "actions_executed": 0,
        "queue_underruns": 0,
        "control_overruns": 0,
        "error": None,
    }
    evaluation_complete = False
    try:
        policy_type = infer_policy_type(args)
        if args.manage_server:
            start_remote_server(args)
            tunnel = start_ssh_tunnel(args)
            server_address = f"127.0.0.1:{args.local_tunnel_port}"
        else:
            server_address = args.server_address
        wait_for_server(server_address, args.connect_timeout_s, tunnel)

        log_eval(
            f"Remote policy: type={policy_type}, inference={args.inference_type}, path={args.policy_path}"
        )
        log_eval(f"Policy server: {server_address}")

        if args.robot_transport == "direct":
            robot_config = AlohaMiniConfig(
                id=args.robot_id,
                robot_model=args.robot_model,
                left_port=args.robot_left_port,
                right_port=args.robot_right_port,
            )
            log_eval("Robot transport: direct hardware access (no AlohaMini Host/ZMQ relay)")
        else:
            robot_config = AlohaMiniClientConfig(
                remote_ip=args.robot_ip,
                id=args.robot_id,
                robot_model=args.robot_model,
            )
            log_eval(f"Robot transport: AlohaMini Host over ZMQ at {args.robot_ip}")
        actions_per_chunk = 1 if args.inference_type == "sync" else args.actions_per_chunk
        if args.inference_type == "sync":
            chunk_size_threshold = 1.0
        elif args.inference_type == "rtc":
            chunk_size_threshold = args.rtc_queue_threshold / actions_per_chunk
        else:
            chunk_size_threshold = args.chunk_size_threshold
        client_config = RobotClientConfig(
            robot=robot_config,
            server_address=server_address,
            policy_type=policy_type,
            pretrained_name_or_path=args.policy_path,
            policy_device=args.policy_device,
            client_device=args.client_device,
            actions_per_chunk=actions_per_chunk,
            chunk_size_threshold=chunk_size_threshold,
            fps=args.fps,
            # RTC replaces the unexecuted suffix with the newest inpainted chunk.
            aggregate_fn_name="latest_only" if args.inference_type == "rtc" else args.aggregate_fn_name,
            inference_type=args.inference_type,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            image_compression_quality=args.image_compression_quality,
            task=args.task_description,
        )
        client = RobotClient(client_config)
        client.policy_config.rename_map = args.policy_rename_map

        robot_observation_processor = None
        dataset_features = None
        if args.record_dataset:
            teleop_action_processor, _, robot_observation_processor = make_default_processors()
            action_dataset_features = aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=client.robot.action_features),
                use_videos=True,
            )
            observation_dataset_features = aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=client.robot.observation_features),
                use_videos=True,
            )
            dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
            dataset = LeRobotDataset.create(
                repo_id=args.dataset_repo_id,
                fps=args.recording_fps,
                features=dataset_features,
                robot_type=client.robot.name,
                use_videos=True,
                image_writer_threads=4,
            )
        else:
            log_eval("Evaluation dataset recording disabled")

        if not client.start():
            raise RuntimeError(f"Could not connect to policy server at {server_address}")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        client.start_barrier.wait()
        action_control_thread = threading.Thread(
            target=run_action_control_loop,
            args=(
                client,
                args.fps,
                args.interpolation_multiplier,
                action_control_stop,
                action_state,
                action_state_lock,
                args.base_snap_speed,
                args.base_snap_deadband,
            ),
            daemon=True,
        )
        action_control_thread.start()
        observation_sender = LatestObservationSender(client, args.task_description)
        observation_sender.start()

        log_eval(f"Starting remote {args.inference_type} evaluation")
        # Observation/inference cadence follows the control FPS. Dataset writing is
        # independently downsampled so camera encoding cannot silently reduce RTC updates.
        loop_interval = 1.0 / args.fps
        recording_interval = 1.0 / args.recording_fps if args.record_dataset else None
        for episode in range(args.num_episodes):
            log_eval(f"Eval episode {episode + 1} of {args.num_episodes}")
            start = time.perf_counter()
            frames_added = 0
            observations_captured = 0
            slow_frames = 0
            next_record_time = start
            with action_state_lock:
                starting_actions = action_state["actions_executed"]
                starting_underruns = action_state["queue_underruns"]
                starting_overruns = action_state["control_overruns"]
            starting_observation_bytes = client.observation_bytes_sent
            starting_observation_seconds = client.observation_send_seconds
            starting_observation_count = client.observation_send_count
            starting_sender_stats = observation_sender.snapshot()
            starting_robot_messages = getattr(client.robot, "observation_messages_received", 0)
            starting_robot_timeouts = getattr(client.robot, "observation_poll_timeouts", 0)

            while time.perf_counter() - start < args.episode_time:
                loop_start = time.perf_counter()
                with action_state_lock:
                    control_error = action_state["error"]
                if control_error is not None:
                    raise RuntimeError("Action control loop failed") from control_error
                sender_error = observation_sender.snapshot()["error"]
                if sender_error is not None:
                    raise RuntimeError("Observation upload loop failed") from sender_error

                observation_raw = client.robot.get_observation()
                observation_received_timestamp = time.time()
                observations_captured += 1
                # Capture and inference remain decoupled. In latest mode every policy sees the newest
                # available visual observation at the target cadence; the client and server both use
                # one-element queues, so slow upload/inference replaces stale pending frames instead
                # of building backlog. queue_gated is retained only for historical async behavior.
                if (
                    args.observation_send_mode == "latest"
                    or args.inference_type in {"sync", "rtc"}
                    or client._ready_to_send_observation()
                ):
                    observation_sender.submit(observation_raw, observation_received_timestamp)

                with action_state_lock:
                    action = action_state["latest_action"]
                now = time.perf_counter()
                should_record = (
                    dataset is not None
                    and action is not None
                    and recording_interval is not None
                    and now >= next_record_time
                )
                if should_record:
                    assert robot_observation_processor is not None
                    assert dataset_features is not None
                    observation_processed = robot_observation_processor(observation_raw)
                    observation_frame = build_dataset_frame(
                        dataset_features, observation_processed, prefix=OBS_STR
                    )
                    action_frame = build_dataset_frame(dataset_features, action, prefix=ACTION)
                    dataset.add_frame({**observation_frame, **action_frame, "task": args.task_description})
                    frames_added += 1
                    next_record_time += recording_interval
                    if next_record_time <= now:
                        next_record_time = now + recording_interval

                elapsed = time.perf_counter() - loop_start
                if elapsed > loop_interval:
                    slow_frames += 1
                elif (sleep_time := loop_interval - elapsed) > 0:
                    precise_sleep(sleep_time)

            episode_elapsed = time.perf_counter() - start
            with action_state_lock:
                actions_executed = action_state["actions_executed"] - starting_actions
                queue_underruns = action_state["queue_underruns"] - starting_underruns
                control_overruns = action_state["control_overruns"] - starting_overruns
            sent_observation_count = client.observation_send_count - starting_observation_count
            sent_observation_bytes = client.observation_bytes_sent - starting_observation_bytes
            sent_observation_seconds = client.observation_send_seconds - starting_observation_seconds
            sender_stats = observation_sender.snapshot()
            observations_submitted = int(sender_stats["submitted"]) - int(starting_sender_stats["submitted"])
            observations_replaced = int(sender_stats["replaced"]) - int(starting_sender_stats["replaced"])
            robot_messages = (
                getattr(client.robot, "observation_messages_received", 0) - starting_robot_messages
            )
            robot_timeouts = getattr(client.robot, "observation_poll_timeouts", 0) - starting_robot_timeouts
            mean_observation_kib = (
                sent_observation_bytes / sent_observation_count / 1024 if sent_observation_count else 0.0
            )
            mean_observation_rpc_ms = (
                sent_observation_seconds / sent_observation_count * 1000 if sent_observation_count else 0.0
            )
            if dataset is not None and frames_added:
                dataset.save_episode()
            elif dataset is not None:
                log_eval("No actions were received; skipping the empty episode")
            recorded_frames = str(frames_added) if dataset is not None else "disabled"
            log_eval(
                f"Episode complete: actions={actions_executed}, recorded_frames={recorded_frames}, "
                f"observations_captured={observations_captured}, observations_submitted={observations_submitted}, "
                f"observations_sent={sent_observation_count}, observations_replaced={observations_replaced}, "
                f"host_messages_received={robot_messages}, host_poll_timeouts={robot_timeouts}, "
                f"slow_frames={slow_frames}, "
                f"control_hz={actions_executed / episode_elapsed:.2f}, "
                f"capture_hz={observations_captured / episode_elapsed:.2f}, "
                f"observation_send_hz={sent_observation_count / episode_elapsed:.2f}, "
                f"queue_underruns={queue_underruns}, control_overruns={control_overruns}, "
                f"mean_observation_kib={mean_observation_kib:.1f}, "
                f"mean_observation_rpc_ms={mean_observation_rpc_ms:.1f}, "
                f"mean_upload_queue_ms={float(sender_stats['mean_queue_delay_ms']):.1f}, "
                f"max_upload_queue_ms={float(sender_stats['max_queue_delay_ms']):.1f}"
            )

        log_eval(f"Remote {args.inference_type} evaluation complete")
        evaluation_complete = True
    finally:
        action_control_stop.set()
        if action_control_thread is not None:
            action_control_thread.join(timeout=5)
        if observation_sender is not None:
            observation_sender.stop()
        if client is not None:
            with suppress(Exception):
                send_base_stop(client)
            with suppress(Exception):
                client.stop()
        if action_receiver_thread is not None:
            action_receiver_thread.join(timeout=5)
        stop_tunnel(tunnel)
        if dataset is not None:
            dataset_finalized = False
            try:
                dataset.finalize()
                dataset_finalized = True
            except Exception as error:
                print(f"Failed to finalize evaluation dataset: {error}", flush=True)
            if args.push_to_hub and evaluation_complete and dataset_finalized:
                dataset.push_to_hub()


if __name__ == "__main__":
    main()
