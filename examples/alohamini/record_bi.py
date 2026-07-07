#!/usr/bin/env python3

from email import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.common.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from datetime import datetime
import argparse
import json
import threading
import time
from pathlib import Path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


class PreviewFrameWriter:
    def __init__(self, preview_dir: str | None, fps: int = 8, quality: int = 70):
        self.preview_dir = Path(preview_dir) if preview_dir else None
        self.period_s = 1.0 / max(int(fps), 1)
        self.quality = int(quality)
        self._last_write_t = 0.0
        self._cv2 = None
        if self.preview_dir is not None:
            self.preview_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, observation: dict) -> None:
        if self.preview_dir is None:
            return
        now = time.monotonic()
        if now - self._last_write_t < self.period_s:
            return
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        for name, value in observation.items():
            if not hasattr(value, "shape") or len(value.shape) != 3:
                continue
            ok, buffer = self._cv2.imencode(
                ".jpg",
                value,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), self.quality],
            )
            if not ok:
                continue
            target = self.preview_dir / f"{name}.jpg"
            tmp = self.preview_dir / f".{name}.jpg.tmp"
            tmp.write_bytes(buffer.tobytes())
            tmp.replace(target)
        self._last_write_t = now


class PhaseMarkerRecorder:
    def __init__(
        self,
        event_file: str | None,
        output_file: str | None,
        dataset_id: str,
        task_description: str,
        fps: int,
    ):
        self.event_file = Path(event_file) if event_file else None
        self.output_file = Path(output_file) if output_file else None
        self.dataset_id = dataset_id
        self.task_description = task_description
        self.fps = fps
        self._event_offset = 0
        self._pending_events: list[dict] = []
        self._episode_markers: list[dict] = []
        self._last_frame: tuple[int, int, float] | None = None
        if self.output_file is not None:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self._append_jsonl(
                {
                    "type": "session_start",
                    "dataset": self.dataset_id,
                    "task": self.task_description,
                    "fps": self.fps,
                    "wall_time_ns": time.time_ns(),
                }
            )

    @property
    def enabled(self) -> bool:
        return self.event_file is not None and self.output_file is not None

    def on_frame(self, episode_index: int, frame_index: int, timestamp: float) -> None:
        if not self.enabled:
            return
        self._last_frame = (episode_index, frame_index, timestamp)
        self._load_new_events()
        if not self._pending_events:
            return
        self._commit_pending_events(episode_index, frame_index, timestamp)

    def _commit_pending_events(self, episode_index: int, frame_index: int, timestamp: float) -> None:
        for event in self._pending_events:
            self._episode_markers.append(
                {
                    "type": "phase_marker",
                    "dataset": self.dataset_id,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "fps": self.fps,
                    "name": event.get("name", ""),
                    "label": event.get("label", ""),
                    "key": event.get("key", ""),
                    "task": self.task_description,
                    "gui_wall_time_ns": event.get("wall_time_ns"),
                    "record_wall_time_ns": time.time_ns(),
                }
            )
        self._pending_events.clear()

    def flush_episode(self) -> None:
        if not self.enabled:
            return
        self._load_new_events()
        if self._pending_events and self._last_frame is not None:
            self._commit_pending_events(*self._last_frame)
        for marker in self._episode_markers:
            self._append_jsonl(marker)
        if self._episode_markers:
            print(f"Phase markers saved: {len(self._episode_markers)} marker(s).")
        self._episode_markers.clear()
        self._pending_events.clear()

    def clear_episode(self) -> None:
        self._load_new_events()
        self._episode_markers.clear()
        self._pending_events.clear()
        self._last_frame = None

    def _load_new_events(self) -> None:
        if self.event_file is None:
            return
        try:
            with self.event_file.open("r", encoding="utf-8") as f:
                f.seek(self._event_offset)
                chunk = f.read()
                self._event_offset = f.tell()
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"Error reading phase marker file {self.event_file}: {exc}")
            return
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid phase marker event ignored: {exc}")
                continue
            if event.get("type") == "phase_marker":
                self._pending_events.append(event)

    def _append_jsonl(self, payload: dict) -> None:
        if self.output_file is None:
            return
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Record episodes with bi-arm teleoperation")
    parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset repo_id, e.g. liyitenga/record_20250914225057")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=60, help="Duration of each episode (seconds)")
    parser.add_argument("--reset_time", type=int, default=10, help="Reset duration between episodes (seconds)")
    parser.add_argument("--task_description", type=str, default="My task description4", help="Task description")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Robot host IP")
    parser.add_argument("--robot_id", type=str, default="lekiwi_host", help="Robot ID")
    parser.add_argument(
        "--robot_model",
        type=str,
        default="alohamini1",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
        help="AlohaMini model. Must match the --robot_model used on the Pi host side.",
    )
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-leader-6dof"],
        help="Leader arm profile selector.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume recording on existing dataset")
    parser.add_argument(
        "--control_file",
        type=str,
        default=None,
        help="Optional file watched for GUI commands: finish, rerecord, stop.",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Optional JSON file watched for GUI reset teleop keys.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to upload the dataset to Hugging Face Hub after recording. Use '--push_to_hub false' to skip upload.",
    )
    parser.add_argument("--preview_dir", type=str, default=None, help="Optional directory for GUI preview JPEGs.")
    parser.add_argument("--preview_fps", type=int, default=8, help="GUI preview frame rate.")
    parser.add_argument("--phase_marker_file", type=str, default=None, help="Optional GUI phase marker event jsonl.")
    parser.add_argument("--phase_marker_output", type=str, default=None, help="Optional sidecar phase marker output jsonl.")

    args = parser.parse_args()

    # === Robot and teleop config ===
    robot_config = LeKiwiClientConfig(
        remote_ip=args.remote_ip,
        id=args.robot_id,
        robot_model=args.robot_model,
    )
    leader_arm_config = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(
            port="/dev/am_arm_leader_left",
            arm_profile=args.arm_profile,
        ),
        right_arm_config=SOLeaderConfig(
            port="/dev/am_arm_leader_right",
            arm_profile=args.arm_profile,
        ),
        id=args.leader_id,
    )
    keyboard_config = KeyboardTeleopConfig()

    robot = LeKiwiClient(robot_config)
    leader_arm = BiSOLeader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    preview_writer = PreviewFrameWriter(args.preview_dir, fps=args.preview_fps)
    phase_markers = PhaseMarkerRecorder(
        event_file=args.phase_marker_file,
        output_file=args.phase_marker_output,
        dataset_id=args.dataset,
        task_description=args.task_description,
        fps=args.fps,
    )

    # === Dataset setup ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        print("Resuming existing dataset:", args.dataset)
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset,
            root=HF_LEROBOT_HOME / args.dataset,
            image_writer_threads=4,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        print(f"Dataset created with id: {dataset.repo_id}")

    print(f"Local dataset path: {dataset.root.resolve()}")

    # === Connect devices ===
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    control_stop = threading.Event()
    manual_finish_wait = threading.Event()
    manual_rerecord_wait = threading.Event()
    restart_requested = threading.Event()
    gui_motion_enabled = threading.Event()
    gui_motion_keys = {"w", "s", "z", "x", "a", "d", "u", "j", "r", "f", " "}

    def watch_control_file():
        if not args.control_file:
            return
        control_path = Path(args.control_file)
        last_raw_command = ""
        while not control_stop.is_set() and not events["stop_recording"]:
            try:
                raw_command = control_path.read_text(encoding="utf-8").strip().lower()
            except FileNotFoundError:
                raw_command = ""
            except Exception as exc:
                print(f"Error reading control file {control_path}: {exc}")
                raw_command = ""

            if raw_command and raw_command != last_raw_command:
                last_raw_command = raw_command
                command = raw_command.split(maxsplit=1)[0]
                if command == "finish":
                    print("Control command received: finish current episode.")
                    events["exit_early"] = True
                elif command == "finish_wait":
                    print("Control command received: finish current episode and wait for manual reset.")
                    restart_requested.clear()
                    manual_finish_wait.set()
                    events["exit_early"] = True
                elif command == "rerecord":
                    print("Control command received: rerecord current episode.")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif command == "rerecord_wait":
                    print("Control command received: discard current episode and wait for manual reset.")
                    restart_requested.clear()
                    manual_rerecord_wait.set()
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif command == "restart":
                    print("Control command received: manual reset complete, restart current episode.")
                    restart_requested.set()
                    events["exit_early"] = True
                elif command == "stop":
                    print("Control command received: stop recording.")
                    events["stop_recording"] = True
                    events["exit_early"] = True
                    restart_requested.set()
            time.sleep(0.1)

    def watch_motion_file():
        if not args.motion_file:
            return
        motion_path = Path(args.motion_file)
        last_stamp = None
        active_keys: set[str] = set()
        while not control_stop.is_set() and not events["stop_recording"]:
            keys: set[str] = set()
            stamp = None
            if gui_motion_enabled.is_set():
                try:
                    payload = json.loads(motion_path.read_text(encoding="utf-8") or "{}")
                    keys = {str(key).lower() for key in payload.get("keys", [])}
                    keys = keys.intersection(gui_motion_keys)
                    stamp = payload.get("stamp")
                except FileNotFoundError:
                    keys = set()
                except Exception as exc:
                    if stamp != last_stamp:
                        print(f"Error reading motion file {motion_path}: {exc}")
                    keys = set()
            if keys != active_keys or stamp != last_stamp:
                for key in gui_motion_keys:
                    keyboard.current_pressed[key] = key in keys
                active_keys = keys
                last_stamp = stamp
            time.sleep(0.03)

    control_thread = None
    if args.control_file:
        control_thread = threading.Thread(target=watch_control_file, daemon=True)
        control_thread.start()
    motion_thread = None
    if args.motion_file:
        motion_thread = threading.Thread(target=watch_motion_file, daemon=True)
        motion_thread.start()
    init_rerun(session_name="lekiwi_record")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")
    recorded_episodes = 0

    def wait_for_gui_restart() -> bool:
        """Allow GUI teleop during manual reset without adding frames to the dataset."""
        events["exit_early"] = False
        gui_motion_enabled.set()
        while not events["stop_recording"] and not restart_requested.is_set():
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=0.5,
                single_task=args.task_description,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                preview_callback=preview_writer,
            )
        gui_motion_enabled.clear()
        for key in gui_motion_keys:
            keyboard.current_pressed[key] = False
        restart_requested.clear()
        events["exit_early"] = False
        return not events["stop_recording"]

    def handle_manual_finish_wait() -> bool:
        nonlocal recorded_episodes

        log_say("Save episode before manual reset")
        dataset.save_episode()
        phase_markers.flush_episode()
        recorded_episodes += 1
        manual_finish_wait.clear()
        return wait_for_gui_restart()

    def handle_manual_rerecord_wait() -> bool:
        log_say("Discard episode and wait for manual reset")
        dataset.clear_episode_buffer()
        phase_markers.clear_episode()
        events["rerecord_episode"] = False
        manual_rerecord_wait.clear()
        return wait_for_gui_restart()

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")

        # === Main record loop ===
        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            preview_callback=preview_writer,
            frame_callback=phase_markers.on_frame,
        )

        if manual_finish_wait.is_set():
            if not handle_manual_finish_wait():
                break
            continue

        if events["rerecord_episode"] and manual_rerecord_wait.is_set():
            if not handle_manual_rerecord_wait():
                break
            continue

        # === Reset environment ===
        if not events["stop_recording"] and (
            (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=args.reset_time,
                single_task=args.task_description,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                preview_callback=preview_writer,
            )

        if manual_finish_wait.is_set():
            if not handle_manual_finish_wait():
                break
            continue

        if events["rerecord_episode"] and manual_rerecord_wait.is_set():
            if not handle_manual_rerecord_wait():
                break
            continue

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            phase_markers.clear_episode()
            continue

        dataset.save_episode()
        phase_markers.flush_episode()
        recorded_episodes += 1

    # === Clean up ===
    log_say("Stop recording")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    if listener is not None:
        listener.stop()
    control_stop.set()
    if control_thread is not None:
        control_thread.join(timeout=1)
    if motion_thread is not None:
        motion_thread.join(timeout=1)
    dataset.finalize()
    print(f"Dataset saved locally at: {dataset.root.resolve()}")
    if args.push_to_hub:
        print(f"Uploading dataset to Hugging Face Hub: {dataset.repo_id}")
        dataset.push_to_hub()
        print(f"Dataset uploaded to: https://huggingface.co/datasets/{dataset.repo_id}")
    else:
        print("Skipping Hugging Face upload because --push_to_hub is false.")


if __name__ == "__main__":
    main()
