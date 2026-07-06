#!/usr/bin/env python3
import argparse
import queue
import select
import sys
import termios
import time
import tty

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from workers.voice_worker import VoiceCommandThread, low_speed_motion_action


KEY_BINDINGS = {
    "w": "forward",
    "s": "backward",
    "z": "left",
    "x": "right",
    "a": "rotate_left",
    "d": "rotate_right",
    "u": "lift_up",
    "j": "lift_down",
    "r": "speed_up",
    "f": "speed_down",
}


def read_terminal_keys() -> list[str]:
    keys = []
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch:
            keys.append(ch)
    return keys


def compact_action_view(action: dict) -> str:
    return (
        f"x={action.get('x.vel', 0): .3f} "
        f"y={action.get('y.vel', 0): .3f} "
        f"theta={action.get('theta.vel', 0): .1f} "
        f"lift_vel={action.get('lift_axis.vel', 0)} "
        f"lift_h={action.get('lift_axis.height_mm', 0): .1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_robot", action="store_true")
    parser.add_argument("--no_leader", action="store_true")
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1")
    parser.add_argument(
        "--robot_model",
        type=str,
        default="alohamini2pro",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
    )
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="am-leader-6dof",
        choices=["so-arm-5dof", "am-leader-6dof"],
    )
    parser.add_argument("--key_hold_s", type=float, default=0.25)
    parser.add_argument("--voice_control", action="store_true", help="Enable microphone voice commands.")
    parser.add_argument("--voice_model", type=str, default="small", help="faster-whisper model name for voice control.")
    parser.add_argument("--voice_device_index", type=str, default=None, help="Optional microphone device index.")
    args = parser.parse_args()

    if not sys.stdin.isatty():
        raise RuntimeError("Terminal-key teleop must be run from an interactive terminal.")

    robot_config = LeKiwiClientConfig(
        remote_ip=args.remote_ip,
        id="my_alohamini",
        robot_model=args.robot_model,
    )
    bi_cfg = BiSOLeaderConfig(
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
    leader = BiSOLeader(bi_cfg)
    robot = LeKiwiClient(robot_config)

    old_tty = termios.tcgetattr(sys.stdin)
    active_until: dict[str, float] = {}
    last_print = 0.0
    last_speed_index = robot.speed_index
    voice_commands: queue.Queue[dict] = queue.Queue()
    voice_thread: VoiceCommandThread | None = None
    voice_motion_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": 0}
    voice_motion_active = False

    try:
        tty.setcbreak(sys.stdin.fileno())

        if not args.no_robot:
            robot.connect()
        if not args.no_leader:
            try:
                leader.connect()
            except Exception as exc:
                print("\nLeader connection failed before teleop started.")
                print("If you only want to test base/lift keyboard control, rerun with: --no_leader")
                print(f"Leader error: {exc}")
                raise
        if not args.no_rerun:
            init_rerun(session_name="lekiwi_teleop")

        print("Terminal-key teleop is active.")
        print("Hold keys: w/s forward/back, z/x strafe, a/d rotate, u/j lift, r/f speed, space stop, q quit.")
        if args.voice_control:
            voice_thread = VoiceCommandThread(
                on_command=voice_commands.put,
                on_log=lambda level, message: print(f"[voice:{level}] {message}", flush=True),
                model_name=args.voice_model,
                device_index=args.voice_device_index,
            )
            voice_thread.start()
            print("Voice control enabled. Say: 前进/后退/左转/右转/上升/下降/停止.")

        while True:
            t0 = time.perf_counter()
            now = time.perf_counter()

            for ch in read_terminal_keys():
                if ch == "\x03" or ch == "q":
                    raise KeyboardInterrupt
                if ch == " ":
                    active_until.clear()
                    continue
                key_char = ch.lower()
                command = KEY_BINDINGS.get(key_char)
                if command:
                    active_until[key_char] = now + args.key_hold_s

            while True:
                try:
                    voice_command = voice_commands.get_nowait()
                except queue.Empty:
                    break
                kind = voice_command.get("kind")
                name = voice_command.get("name")
                text = voice_command.get("text", "")
                if kind == "emergency_stop":
                    active_until.clear()
                    voice_motion_active = False
                    voice_motion_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": 0}
                    print(f"[voice] emergency_stop: {text}", flush=True)
                elif kind == "motion":
                    voice_motion_action = low_speed_motion_action(name)
                    voice_motion_active = True
                    print(f"[voice] {name}: {text}", flush=True)
                elif kind == "record":
                    print(f"[voice] ignored recorder command in terminal teleop: {name} ({text})", flush=True)

            active_until = {cmd: expiry for cmd, expiry in active_until.items() if expiry > now}
            keyboard_keys = {key: None for key in active_until}

            observation = robot.get_observation() if not args.no_robot else {}
            arm_actions = leader.get_action() if not args.no_leader else {}
            arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)
            lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)
            action = {**arm_actions, **base_action, **lift_action}
            if voice_motion_active and not keyboard_keys:
                action.update(voice_motion_action)

            if not args.no_rerun:
                log_rerun_data(observation, action)
            if not args.no_robot:
                robot.send_action(action)

            nonzero_motion = (
                abs(float(action.get("x.vel", 0))) > 1e-9
                or abs(float(action.get("y.vel", 0))) > 1e-9
                or abs(float(action.get("theta.vel", 0))) > 1e-9
                or abs(float(action.get("lift_axis.vel", 0))) > 1e-9
            )
            speed_changed = robot.speed_index != last_speed_index
            if nonzero_motion or speed_changed or now - last_print > 1.0:
                labels = [KEY_BINDINGS.get(key, key) for key in sorted(active_until)]
                print(f"[keys={','.join(labels) or '-'}] {compact_action_view(action)}")
                sys.stdout.flush()
                last_print = now
                last_speed_index = robot.speed_index

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nStopping teleop.")
    finally:
        if voice_thread is not None:
            voice_thread.stop()
            voice_thread.join(timeout=1.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        try:
            stop_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.vel": 0}
            if not args.no_robot:
                robot.send_action(stop_action)
        except Exception as exc:
            print(f"Stop action failed: {exc}")
        try:
            if not args.no_leader:
                leader.disconnect()
        except Exception:
            pass
        try:
            if not args.no_robot:
                robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
