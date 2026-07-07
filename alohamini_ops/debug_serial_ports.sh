#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

resolve_local_repo() {
  if [[ -d "$LOCAL_REPO/src/lerobot" ]]; then
    echo "$LOCAL_REPO"
  elif [[ "$LOCAL_REPO" != /* && -d "$OPS_DIR/../$LOCAL_REPO/src/lerobot" ]]; then
    (cd "$OPS_DIR/.." && cd "$LOCAL_REPO" && pwd)
  elif [[ -d "$OPS_DIR/../src/lerobot" ]]; then
    (cd "$OPS_DIR/.." && pwd)
  elif [[ -d "$OPS_DIR/../lerobot_alohamini/src/lerobot" ]]; then
    (cd "$OPS_DIR/../lerobot_alohamini" && pwd)
  else
    echo "$LOCAL_REPO"
  fi
}

LOCAL_REPO_RESOLVED="$(resolve_local_repo)"

print_expectations() {
  cat <<'MSG'
== Expected AlohaMini serial layout ==
Local PC:
  /dev/am_arm_leader_left   -> Leader left arm, usually servo IDs 1-7
  /dev/am_arm_leader_right  -> Leader right arm, usually servo IDs 1-7
Raspberry Pi:
  /dev/am_arm_follower_left -> Follower left arm IDs 1-7 + base 8/9/10 + lift 11
  /dev/am_arm_follower_right-> Follower right arm IDs 1-7

This tool is read-only. It does not install udev rules and does not change ports.
If /dev/am_arm_* links look wrong, use this output to fix udev rules or cabling manually.
MSG
}

list_devices() {
  local title="$1"
  shift

  echo
  echo "== $title =="
  echo "-- user/groups --"
  id || true
  echo
  echo "-- device links and candidates --"
  for path in "$@"; do
    ls -l "$path" 2>/dev/null || true
  done
  echo
  echo "-- serial by-id real paths --"
  for path in /dev/serial/by-id/*; do
    [[ -e "$path" ]] || continue
    printf "%s -> %s\n" "$path" "$(readlink -f "$path" 2>/dev/null || true)"
  done
}

collect_candidates() {
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    printf "%s\n" "$path"
  done
}

scan_port() {
  local port="$1"

  echo
  echo "-- scan $port --"
  if [[ ! -e "$port" ]]; then
    echo "MISSING: $port"
    return 0
  fi
  if [[ ! -r "$port" || ! -w "$port" ]]; then
    echo "PERMISSION: $port is not readable/writable by current user"
    echo "groups: $(id -nG)"
    return 0
  fi

  local exit_code=0
  timeout 6 python examples/debug/motors.py get_motors_states --port "$port" || exit_code=$?
  if [[ "$exit_code" -eq 124 ]]; then
    echo "TIMEOUT: stopped scan after 6s. If IDs were printed above, the port is usable."
  elif [[ "$exit_code" -ne 0 ]]; then
    echo "FAILED: scan exited with code $exit_code. Port may be busy or not a Feetech bus."
  fi
}

run_local_scan() {
  echo
  echo "== Local PC serial debug =="
  echo "Local repo: $LOCAL_REPO_RESOLVED"
  if [[ ! -d "$LOCAL_REPO_RESOLVED/src/lerobot" ]]; then
    echo "ERROR: LOCAL_REPO does not point to a LeRobot checkout: $LOCAL_REPO_RESOLVED"
    return 0
  fi

  source "$CONDA_INIT_LOCAL"
  conda activate "$CONDA_ENV"
  cd "$LOCAL_REPO_RESOLVED"

  list_devices "Local device inventory" \
    /dev/am_arm_leader_left /dev/am_arm_leader_right \
    /dev/ttyACM* /dev/ttyUSB* /dev/serial/by-id/*

  mapfile -t ports < <(collect_candidates /dev/am_arm_leader_left /dev/am_arm_leader_right /dev/ttyACM* /dev/ttyUSB*)
  if [[ "${#ports[@]}" -eq 0 ]]; then
    echo
    echo "未发现候选串口: /dev/am_arm_leader_* /dev/ttyACM* /dev/ttyUSB*"
    return 0
  fi

  echo
  echo "== Local read-only servo scans =="
  for port in "${ports[@]}"; do
    scan_port "$port"
  done
}

run_pi_scan() {
  echo
  echo "== Raspberry Pi serial debug =="
  echo "Target: $PI_USER@$PI_HOST"
  echo "If Host is running, follower ports may be busy. This script will not stop Host."

  if ! ssh "$PI_USER@$PI_HOST" \
    "CONDA_INIT_PI=$(printf '%q' "$CONDA_INIT_PI") CONDA_ENV=$(printf '%q' "$CONDA_ENV") PI_REPO=$(printf '%q' "$PI_REPO") bash -s" <<'REMOTE'
set -euo pipefail

list_devices() {
  local title="$1"
  shift

  echo
  echo "== $title =="
  echo "-- user/groups --"
  id || true
  echo
  echo "-- device links and candidates --"
  for path in "$@"; do
    ls -l "$path" 2>/dev/null || true
  done
  echo
  echo "-- serial by-id real paths --"
  for path in /dev/serial/by-id/*; do
    [[ -e "$path" ]] || continue
    printf "%s -> %s\n" "$path" "$(readlink -f "$path" 2>/dev/null || true)"
  done
}

collect_candidates() {
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    printf "%s\n" "$path"
  done
}

scan_port() {
  local port="$1"

  echo
  echo "-- scan $port --"
  if [[ ! -e "$port" ]]; then
    echo "MISSING: $port"
    return 0
  fi
  if [[ ! -r "$port" || ! -w "$port" ]]; then
    echo "PERMISSION: $port is not readable/writable by current user"
    echo "groups: $(id -nG)"
    return 0
  fi

  local exit_code=0
  timeout 6 python examples/debug/motors.py get_motors_states --port "$port" || exit_code=$?
  if [[ "$exit_code" -eq 124 ]]; then
    echo "TIMEOUT: stopped scan after 6s. If IDs were printed above, the port is usable."
  elif [[ "$exit_code" -ne 0 ]]; then
    echo "FAILED: scan exited with code $exit_code. Port may be busy or not a Feetech bus."
  fi
}

echo "-- host process check --"
pgrep -af '[p]ython -m lerobot.robots.alohamini.lekiwi_host' || true

list_devices "Pi device inventory" \
  /dev/am_arm_follower_left /dev/am_arm_follower_right \
  /dev/ttyACM* /dev/ttyUSB* /dev/serial/by-id/*

source "$CONDA_INIT_PI"
conda activate "$CONDA_ENV"
cd "$PI_REPO"

mapfile -t ports < <(collect_candidates /dev/am_arm_follower_left /dev/am_arm_follower_right /dev/ttyACM* /dev/ttyUSB*)
if [[ "${#ports[@]}" -eq 0 ]]; then
  echo
  echo "未发现候选串口: /dev/am_arm_follower_* /dev/ttyACM* /dev/ttyUSB*"
  exit 0
fi

echo
echo "== Pi read-only servo scans =="
for port in "${ports[@]}"; do
  scan_port "$port"
done
REMOTE
  then
    echo
    echo "FAILED: cannot run Pi serial debug over SSH. Check PI_HOST/PI_USER and network."
  fi
}

print_expectations
run_local_scan
run_pi_scan
