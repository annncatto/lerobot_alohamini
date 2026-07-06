#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$OPS_DIR/.." && pwd)"
CONFIG_ENV="$OPS_DIR/config.env"

usage() {
  cat <<'MSG'
Usage:
  alohamini_ops/init_customer_env.sh [--pi-user USER] [--pi-host HOST] [--conda-env NAME]

This script prepares alohamini_ops/config.env after a fresh git clone.
It preserves unrelated config keys and updates only machine-specific paths
and the Raspberry Pi connection target.

Examples:
  alohamini_ops/init_customer_env.sh
  alohamini_ops/init_customer_env.sh --pi-user pi5 --pi-host 192.168.0.24
MSG
}

read_config_value() {
  local key="$1"
  if [ -f "$CONFIG_ENV" ]; then
    awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2); exit}' "$CONFIG_ENV"
  fi
}

find_conda_init() {
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/opt/anaconda3/etc/profile.d/conda.sh"
  )
  for path in "${candidates[@]}"; do
    if [ -f "$path" ]; then
      echo "$path"
      return 0
    fi
  done
  echo "$HOME/miniconda3/etc/profile.d/conda.sh"
}

discover_pi_candidates() {
  {
    ip neigh show 2>/dev/null | awk '/lladdr/ && $1 ~ /^[0-9]+\./ {print $1}'
    arp -a 2>/dev/null | awk 'match($0, /\(([0-9.]+)\)/) {print substr($0, RSTART + 1, RLENGTH - 2)}'
  } | awk '!seen[$0]++'
}

find_local_repo() {
  if [ -d "$ROOT_DIR/src/lerobot" ]; then
    echo "$ROOT_DIR"
  elif [ -d "$ROOT_DIR/lerobot_alohamini/src/lerobot" ]; then
    echo "$ROOT_DIR/lerobot_alohamini"
  else
    echo "$ROOT_DIR"
  fi
}

choose_pi_host() {
  local default_host="$1"
  local candidates
  candidates="$(discover_pi_candidates || true)"

  if [ -n "$default_host" ]; then
    echo "$default_host"
    return 0
  fi

  if [ ! -t 0 ]; then
    echo ""
    return 0
  fi

  echo "== Raspberry Pi host discovery =="
  if [ -n "$candidates" ]; then
    echo "Network candidates:"
    printf '  %s\n' $candidates
  else
    echo "No LAN candidates found from ip neigh / arp."
  fi
  echo
  read -r -p "Enter Raspberry Pi IP/hostname: " host
  echo "$host"
}

set_config_value() {
  local key="$1"
  local value="$2"
  local tmp
  tmp="$(mktemp)"

  if [ -f "$CONFIG_ENV" ]; then
    awk -v key="$key" -v value="$value" '
      BEGIN { done = 0 }
      $0 ~ "^[[:space:]]*" key "=" {
        print key "=" value
        done = 1
        next
      }
      { print }
      END {
        if (!done) {
          print key "=" value
        }
      }
    ' "$CONFIG_ENV" > "$tmp"
  else
    printf '%s=%s\n' "$key" "$value" > "$tmp"
  fi

  mv "$tmp" "$CONFIG_ENV"
}

PI_USER_ARG=""
PI_HOST_ARG=""
CONDA_ENV_ARG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pi-user)
      PI_USER_ARG="${2:-}"
      shift 2
      ;;
    --pi-host)
      PI_HOST_ARG="${2:-}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_ARG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

existing_pi_user="$(read_config_value PI_USER || true)"
existing_pi_host="$(read_config_value PI_HOST || true)"
existing_conda_env="$(read_config_value CONDA_ENV || true)"

PI_USER="${PI_USER_ARG:-${ALOHAMINI_PI_USER:-${existing_pi_user:-pi5}}}"
PI_HOST="${PI_HOST_ARG:-${ALOHAMINI_PI_HOST:-}}"
if [ -z "$PI_HOST" ]; then
  PI_HOST="$(choose_pi_host "$existing_pi_host")"
fi
CONDA_ENV="${CONDA_ENV_ARG:-${existing_conda_env:-lerobot_alohamini}}"

if [ -z "$PI_USER" ]; then
  echo "ERROR: PI_USER is empty."
  exit 1
fi
if [ -z "$PI_HOST" ]; then
  echo "ERROR: PI_HOST is empty. Re-run with --pi-host, for example:"
  echo "  $0 --pi-host 192.168.0.24"
  exit 1
fi

if [ -f "$CONFIG_ENV" ]; then
  backup="$CONFIG_ENV.bak.$(date +%Y%m%d_%H%M%S)"
  cp "$CONFIG_ENV" "$backup"
  echo "Backup: $backup"
fi

LOCAL_REPO="$(find_local_repo)"
CONDA_INIT_LOCAL="$(find_conda_init)"

set_config_value PI_USER "$PI_USER"
set_config_value PI_HOST "$PI_HOST"
set_config_value LOCAL_REPO "$LOCAL_REPO"
set_config_value PI_REPO "/home/$PI_USER/lerobot_alohamini"
set_config_value CONDA_INIT_LOCAL "$CONDA_INIT_LOCAL"
set_config_value CONDA_INIT_PI "/home/$PI_USER/miniconda3/etc/profile.d/conda.sh"
set_config_value CONDA_ENV "$CONDA_ENV"
set_config_value PI_LOG_DIR "/home/$PI_USER/alohamini_logs"
set_config_value PI_HOST_LOG "/home/$PI_USER/alohamini_logs/lekiwi_host.log"
set_config_value ALOHAMINI_DATASET_HOME "$ROOT_DIR/datasets/lerobot"
set_config_value ALOHAMINI_CALIBRATION_HOME "$HOME/.cache/huggingface/lerobot/calibration"

echo
echo "AlohaMini customer config is ready:"
echo "  config: $CONFIG_ENV"
echo "  pi:     $PI_USER@$PI_HOST"
echo "  repo:   $LOCAL_REPO"
echo "  conda:  $CONDA_ENV ($CONDA_INIT_LOCAL)"
echo
echo "Next:"
echo "  $OPS_DIR/setup_env.sh"
echo "  $OPS_DIR/start_gui.sh"
