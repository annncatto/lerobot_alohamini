#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ALOHAMINI_CONDA_ENV:-lerobot_alohamini}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_DIR="$REPO_DIR/alohamini_ops"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
INSTALL_VOICE=0
INIT_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pi-user|--pi-host|--conda-env)
      INIT_ARGS+=("$1" "${2:-}")
      if [ "$1" = "--conda-env" ]; then
        ENV_NAME="${2:-$ENV_NAME}"
      fi
      shift 2
      ;;
    --voice)
      INSTALL_VOICE=1
      shift
      ;;
    -h|--help)
      cat <<'MSG'
Usage:
  ./setup_ubuntu_env.sh [--pi-user USER] [--pi-host HOST] [--conda-env NAME] [--voice]

Creates/uses the local conda environment, installs AlohaMini dependencies,
refreshes alohamini_ops/config.env, and installs GUI dependencies.
Use --voice only when microphone voice control is needed.
MSG
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[AlohaMini] Repo: $REPO_DIR"
echo "[AlohaMini] Conda env: $ENV_NAME"

if [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
  echo "[AlohaMini] Miniconda not found, installing to $CONDA_DIR"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$INSTALLER" "$MINICONDA_URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$MINICONDA_URL" -o "$INSTALLER"
  else
    echo "[AlohaMini] Need wget or curl to download Miniconda." >&2
    exit 1
  fi
  bash "$INSTALLER" -b -p "$CONDA_DIR"
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[AlohaMini] Creating conda env $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.12 -y
fi

conda activate "$ENV_NAME"

cd "$REPO_DIR"
python -m pip install --upgrade pip
python -m pip install -e ".[lekiwi,aloha,training,core_scripts,dataset_viz]"

"$OPS_DIR/init_customer_env.sh" "${INIT_ARGS[@]}"
if [ "$INSTALL_VOICE" = "1" ]; then
  "$OPS_DIR/setup_env.sh" --voice
else
  "$OPS_DIR/setup_env.sh"
fi

echo
echo "[AlohaMini] Ubuntu environment is ready."
echo "Run:"
echo "  conda activate $ENV_NAME"
echo "  cd $REPO_DIR"
