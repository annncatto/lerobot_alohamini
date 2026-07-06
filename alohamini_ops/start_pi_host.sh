#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

ssh "$PI_USER@$PI_HOST" "mkdir -p '$PI_LOG_DIR'"

if ssh "$PI_USER@$PI_HOST" "pgrep -f '[p]ython -m lerobot.robots.alohamini.lekiwi_host' >/dev/null"; then
  echo "Pi host is already running."
  ssh "$PI_USER@$PI_HOST" "pgrep -af '[p]ython -m lerobot.robots.alohamini.lekiwi_host' || true"
  exit 0
fi

ssh "$PI_USER@$PI_HOST" "
  cd '$PI_REPO' &&
  nohup bash -lc \"printf '\\n\\n\\n\\n' | bash -lc 'source $CONDA_INIT_PI && conda activate $CONDA_ENV && export ALOHAMINI_CAMERAS=\\\"${ALOHAMINI_CAMERAS:-forward,wrist_right}\\\" && python -m lerobot.robots.alohamini.lekiwi_host --robot_model $ROBOT_MODEL'\" \
    > '$PI_HOST_LOG' 2>&1 < /dev/null &
"

sleep 8

if ssh "$PI_USER@$PI_HOST" "pgrep -f '[p]ython -m lerobot.robots.alohamini.lekiwi_host' >/dev/null"; then
  echo "Pi host started."
  ssh "$PI_USER@$PI_HOST" "pgrep -af '[p]ython -m lerobot.robots.alohamini.lekiwi_host' || true; tail -50 '$PI_HOST_LOG'"
else
  echo "Pi host failed to start. Log:"
  ssh "$PI_USER@$PI_HOST" "tail -120 '$PI_HOST_LOG' 2>/dev/null || true"
  exit 1
fi
