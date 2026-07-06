#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

echo "== Raspberry Pi follower/base servo check =="
echo "Target: $PI_USER@$PI_HOST"
echo "Time: $(date '+%F %T')"
echo

ssh "$PI_USER@$PI_HOST" "source '$CONDA_INIT_PI' && conda activate '$CONDA_ENV' && cd '$PI_REPO' && \
  echo '-- device links --' && \
  ls -l /dev/am_arm_follower_left /dev/am_arm_follower_right /dev/ttyACM* 2>/dev/null || true && \
  echo && \
  echo '-- follower_left + base/lift bus: /dev/am_arm_follower_left --' && \
  timeout 12 python examples/debug/motors.py get_motors_states --port /dev/am_arm_follower_left && \
  echo && \
  echo '-- follower_right: /dev/am_arm_follower_right --' && \
  timeout 12 python examples/debug/motors.py get_motors_states --port /dev/am_arm_follower_right"
