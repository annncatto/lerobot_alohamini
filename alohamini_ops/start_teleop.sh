#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

NO_LEADER=0
for arg in "$@"; do
  if [[ "$arg" == "--no_leader" ]]; then
    NO_LEADER=1
  fi
done

if [[ "$NO_LEADER" == "0" && (! -e /dev/am_arm_leader_left || ! -e /dev/am_arm_leader_right) ]]; then
  echo "Leader serial links are missing: /dev/am_arm_leader_left or /dev/am_arm_leader_right"
  echo "Check USB connection and udev mapping before starting teleop."
  exit 1
fi

if [[ "$NO_LEADER" == "0" && (! -r /dev/am_arm_leader_left || ! -r /dev/am_arm_leader_right) ]]; then
  echo "Leader serial links exist but are not readable by this shell."
  echo "Current groups: $(id -nG)"
  echo "Expected group: dialout. Log out/in if dialout was added recently."
  exit 1
fi

source "$CONDA_INIT_LOCAL"
conda activate "$CONDA_ENV"
cd "$LOCAL_REPO"

echo "Starting teleop. Keep this terminal focused for WASD/ZSXAD/UJ keys."
echo "Full log: $LOCAL_TELEOP_LOG"

ALOHAMINI_CAMERAS="${ALOHAMINI_CAMERAS:-forward,wrist_right}" \
stdbuf -oL -eL python "$OPS_DIR/teleoperate_bi_terminal_keys.py" \
  --remote_ip "$PI_HOST" \
  --robot_model "$ROBOT_MODEL" \
  --leader_id "$LEADER_ID" \
  --arm_profile "$ARM_PROFILE" "$@" 2>&1 \
  | tee "$LOCAL_TELEOP_LOG" \
  | awk '
      /^\[fps=.*Sent action/ {
        n += 1
        if (n % 30 == 1) print
        fflush()
        next
      }
      { print; fflush() }
    '
