#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"

INSTALL_VOICE=0
for arg in "$@"; do
  case "$arg" in
    --voice)
      INSTALL_VOICE=1
      ;;
    -h|--help)
      cat <<'MSG'
Usage:
  alohamini_ops/setup_env.sh [--voice]

Default installs/checks only the basic GUI dependencies.
Use --voice only when enabling microphone voice control.
MSG
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $arg"
      exit 1
      ;;
  esac
done

source "$CONDA_INIT_LOCAL"
conda activate "$CONDA_ENV"

if [ ! -d "$LOCAL_REPO/src/lerobot" ]; then
  echo "ERROR: LOCAL_REPO does not point to a LeRobot checkout:"
  echo "  LOCAL_REPO=$LOCAL_REPO"
  echo
  echo "Copy or clone lerobot_alohamini on this computer, then edit:"
  echo "  $OPS_DIR/config.env"
  echo "Set LOCAL_REPO to the new absolute path."
  exit 1
fi

export PYTHONPATH="$OPS_DIR:$LOCAL_REPO/src:${PYTHONPATH:-}"
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

echo "== Conda environment =="
echo "CONDA_PREFIX=$CONDA_PREFIX"
python -V
python -m pip --version

echo
echo "== Installing GUI dependencies =="
python -m pip install -r "$OPS_DIR/requirements-gui.txt"
if [[ "$INSTALL_VOICE" == "1" ]]; then
  echo
  echo "== Installing optional voice-control dependencies =="
  python -m pip install -r "$OPS_DIR/requirements-voice.txt"
else
  echo
  echo "== Skipping optional voice-control dependencies =="
  echo "Run '$OPS_DIR/setup_env.sh --voice' only if you need microphone voice control."
fi

echo
echo "== Verifying runtime imports =="
python - <<'PY'
checks = [
    "PyQt6",
    "numpy",
    "zmq",
    "cv2",
    "lerobot",
]

failed = False
for name in checks:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "")
        print(f"{name}: OK {version}")
    except Exception as exc:
        failed = True
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")

if failed:
    raise SystemExit(1)
PY

if [[ "$INSTALL_VOICE" == "1" ]]; then
  echo
  echo "== Verifying optional voice imports =="
  python - <<'PY'
checks = ["sounddevice", "faster_whisper"]
failed = False
for name in checks:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "")
        print(f"{name}: OK {version}")
    except Exception as exc:
        failed = True
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit(1)
PY
fi

echo
echo "== Qt platform check =="
if ! python - <<'PY'
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qt_compat import QApplication
app = QApplication([])
print("Qt: OK")
PY
then
  cat <<'MSG'
Qt import succeeded poorly or Qt platform plugins are missing.
On Ubuntu, install the common xcb dependencies, then rerun setup:

  sudo apt-get update
  sudo apt-get install -y libxcb-cursor0 libxcb-cursor-dev libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0

If the error mentions a wrong plugin path, make sure you start with alohamini_ops/start_gui.sh.
MSG
  exit 1
fi

echo
echo "GUI environment is ready."
echo "Start GUI with:"
echo "  $OPS_DIR/start_gui.sh"
