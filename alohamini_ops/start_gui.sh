#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"

source "$CONDA_INIT_LOCAL"
conda activate "$CONDA_ENV"
cd "$OPS_DIR"

if [ ! -d "$LOCAL_REPO/src/lerobot" ]; then
  echo "ERROR: LOCAL_REPO does not point to a LeRobot checkout:"
  echo "  LOCAL_REPO=$LOCAL_REPO"
  echo "Edit $OPS_DIR/config.env on this computer."
  exit 1
fi

export PYTHONPATH="$LOCAL_REPO/src:${PYTHONPATH:-}"
export QT_LOGGING_RULES="${QT_LOGGING_RULES:-qt.qpa.theme.gnome.warning=false}"
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if (importlib.util.find_spec("PyQt6") or importlib.util.find_spec("PySide6")) else 1)
PY
then
  echo "Qt binding is missing in conda env '$CONDA_ENV'."
  echo "Run:"
  echo "  $OPS_DIR/setup_env.sh"
  exit 1
fi

if ! QT_QPA_PLATFORM=offscreen python - <<'PY'
from qt_compat import QApplication
app = QApplication([])
PY
then
  cat <<'MSG'
Qt platform plugin check failed.
On Ubuntu, install:
  sudo apt-get update
  sudo apt-get install -y libxcb-cursor0 libxcb-cursor-dev libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0
Then run alohamini_ops/setup_env.sh again.
MSG
  exit 1
fi

exec python main.py
