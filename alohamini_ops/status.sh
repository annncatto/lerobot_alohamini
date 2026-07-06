#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

echo "== Local leader devices =="
ls -l /dev/am_arm_leader_left /dev/am_arm_leader_right /dev/ttyACM* 2>/dev/null || true

echo
echo "== Local teleop processes =="
pgrep -af "examples/alohamini/teleoperate_bi.py|teleoperate_bi.py" || true

echo
echo "== Pi follower devices =="
ssh "$PI_USER@$PI_HOST" "ls -l /dev/am_arm_follower_left /dev/am_arm_follower_right /dev/ttyACM* 2>/dev/null || true"

echo
echo "== Pi host processes =="
ssh "$PI_USER@$PI_HOST" "pgrep -af '[p]ython -m lerobot.robots.alohamini.lekiwi_host' || true"

echo
echo "== Pi host log tail =="
ssh "$PI_USER@$PI_HOST" "tail -40 '$PI_HOST_LOG' 2>/dev/null || true"
