#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"

source "$CONDA_INIT_LOCAL"
conda activate "$CONDA_ENV"
cd "$LOCAL_REPO"

echo "== Local Leader servo check =="
echo "Time: $(date '+%F %T')"
echo

check_port() {
  local label="$1"
  local port="$2"

  echo "-- $label: $port --"
  if [[ ! -e "$port" ]]; then
    echo "MISSING: $port does not exist"
    echo
    return 1
  fi
  if [[ ! -r "$port" || ! -w "$port" ]]; then
    echo "PERMISSION: $port is not readable/writable by current shell"
    echo "groups: $(id -nG)"
    echo
    return 1
  fi

  local scan_log
  scan_log="$(mktemp)"
  timeout 12 python examples/debug/motors.py get_motors_states --port "$port" | tee "$scan_log" || {
    rm -f "$scan_log"
    echo "FAILED: motor state scan failed on $port"
    echo
    return 1
  }
  if grep -q "No motors found" "$scan_log"; then
    rm -f "$scan_log"
    echo "FAILED: no servo IDs found on $port"
    echo
    return 1
  fi
  rm -f "$scan_log"
  echo
}

status=0
check_port "leader_left" /dev/am_arm_leader_left || status=1
check_port "leader_right" /dev/am_arm_leader_right || status=1
exit "$status"
