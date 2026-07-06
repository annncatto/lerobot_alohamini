#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"

if [[ ! -e /dev/am_arm_leader_left || ! -e /dev/am_arm_leader_right ]]; then
  echo "Leader serial links are missing: /dev/am_arm_leader_left or /dev/am_arm_leader_right"
  echo "Check USB connection and udev mapping before calibration."
  exit 1
fi

if [[ ! -r /dev/am_arm_leader_left || ! -r /dev/am_arm_leader_right ]]; then
  echo "Leader serial links exist but are not readable by this shell."
  echo "Current groups: $(id -nG)"
  echo "Expected group: dialout. Log out/in if dialout was added recently."
  exit 1
fi

source "$CONDA_INIT_LOCAL"
conda activate "$CONDA_ENV"
cd "$LOCAL_REPO"

printf '\n\n' | lerobot-calibrate \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/am_arm_leader_left \
  --teleop.right_arm_config.port=/dev/am_arm_leader_right \
  --teleop.id="$LEADER_ID" \
  --teleop.left_arm_config.arm_profile="$ARM_PROFILE" \
  --teleop.right_arm_config.arm_profile="$ARM_PROFILE"
