#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

ssh "$PI_USER@$PI_HOST" "pkill -f '[p]ython -m lerobot.robots.alohamini.lekiwi_host' || true"
sleep 1
ssh "$PI_USER@$PI_HOST" "pgrep -af '[p]ython -m lerobot.robots.alohamini.lekiwi_host' || true"
echo "Pi host stop command sent."
