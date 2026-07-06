#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

STEPS="${1:-10}"
STEP_MM="${2:-20}"
DWELL_S="${3:-1.0}"

if ssh "$PI_USER@$PI_HOST" "pgrep -f '[p]ython -m lerobot.robots.alohamini.lekiwi_host' >/dev/null"; then
  echo "Pi host is running. Stop it before lift-axis diagnosis:"
  echo "  $OPS_DIR/stop_pi_host.sh"
  echo
  echo "This test needs exclusive access to the follower left bus."
  exit 1
fi

ssh "$PI_USER@$PI_HOST" \
  "cd '$PI_REPO' && source '$CONDA_INIT_PI' && conda activate '$CONDA_ENV' && ALOHAMINI_ROBOT_MODEL='$ROBOT_MODEL' ALOHAMINI_LIFT_STEPS='$STEPS' ALOHAMINI_LIFT_STEP_MM='$STEP_MM' ALOHAMINI_LIFT_DWELL_S='$DWELL_S' python -" <<'PY'
import os
import time
from statistics import mean

from lerobot.robots.alohamini.config_lekiwi import LeKiwiConfig
from lerobot.robots.alohamini.lekiwi import LeKiwi


robot_model = os.environ.get("ALOHAMINI_ROBOT_MODEL", "alohamini2pro")
steps = int(os.environ.get("ALOHAMINI_LIFT_STEPS", "10"))
step_mm = float(os.environ.get("ALOHAMINI_LIFT_STEP_MM", "20"))
dwell_s = float(os.environ.get("ALOHAMINI_LIFT_DWELL_S", "1.0"))


def safe_read(bus, item, name, default=None):
    try:
        return bus.read(item, name, normalize=False)
    except Exception:
        return default


cfg = LeKiwiConfig()
cfg.id = "AlohaMiniRobot"
cfg.robot_model = robot_model
cfg.no_follower = True
cfg.cameras = {}
robot = LeKiwi(cfg)

name = robot.lift.cfg.name
samples = []

print("== Lift axis diagnosis ==")
print(f"robot_model={robot_model}")
print(f"relative steps={steps}, step_mm={step_mm}, dwell_s={dwell_s}")
print("NOTE: relative height starts at 0 for this diagnosis; it does not re-home the lift.")

try:
    robot.left_bus.connect()
    robot.lift.configure()
    start_pos = safe_read(robot.left_bus, "Present_Position", name)
    start_h = robot.lift.get_height_mm()
    print(f"start: height={start_h:.1f}mm raw_pos={start_pos}")

    prev_h = start_h
    stalled_steps = 0
    high_current_steps = 0

    target = start_h
    for i in range(1, steps + 1):
        target = max(target + step_mm, robot.lift.get_height_mm() + step_mm)
        t_end = time.time() + dwell_s
        step_samples = []
        robot.lift.apply_action({f"{name}.height_mm": target})

        while time.time() < t_end:
            h = robot.lift.get_height_mm()
            raw_pos = safe_read(robot.left_bus, "Present_Position", name)
            raw_vel = safe_read(robot.left_bus, "Present_Velocity", name, 0)
            raw_cur = safe_read(robot.left_bus, "Present_Current", name, 0)
            cur_ma = float(raw_cur or 0) * 6.5
            step_samples.append((h, raw_pos, raw_vel, cur_ma))
            time.sleep(0.1)

        robot.lift.apply_action({f"{name}.height_mm": target})
        h = robot.lift.get_height_mm()
        raw_pos = safe_read(robot.left_bus, "Present_Position", name)
        raw_vel = safe_read(robot.left_bus, "Present_Velocity", name, 0)
        raw_cur = safe_read(robot.left_bus, "Present_Current", name, 0)
        cur_ma = float(raw_cur or 0) * 6.5
        moved = h - prev_h
        avg_cur = mean([s[3] for s in step_samples]) if step_samples else cur_ma
        samples.append((target, h, moved, raw_pos, raw_vel, cur_ma, avg_cur))

        print(
            f"step {i:02d}: target={target:6.1f}mm "
            f"height={h:7.1f}mm moved={moved:6.1f}mm "
            f"raw_pos={raw_pos} raw_vel={raw_vel} current={cur_ma:7.1f}mA avg_current={avg_cur:7.1f}mA"
        )

        if moved < max(1.0, step_mm * 0.15):
            stalled_steps += 1
        else:
            stalled_steps = 0
        if avg_cur > 1200:
            high_current_steps += 1
        if stalled_steps >= 2:
            print("DIAG: height stopped increasing for 2 consecutive steps.")
            if high_current_steps:
                print("DIAG: current is elevated; likely mechanical resistance, hard limit, or power issue.")
            else:
                print("DIAG: current is not elevated; likely command/measurement/soft-limit issue.")
            break

        prev_h = h

    final_h = robot.lift.get_height_mm()
    print(f"final: relative_height={final_h:.1f}mm")
    print(f"configured soft_max_mm={robot.lift.cfg.soft_max_mm}, v_max={robot.lift.cfg.v_max}, kp_vel={robot.lift.cfg.kp_vel}")

finally:
    try:
        robot.lift.stop()
    except Exception as exc:
        print(f"WARN: lift stop failed: {exc}")
    try:
        robot.left_bus.disconnect(disable_torque=False)
    except TypeError:
        try:
            robot.left_bus.disconnect()
        except Exception:
            pass
    except Exception:
        pass
PY
