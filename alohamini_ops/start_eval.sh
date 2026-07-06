#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

DATASET_HOME="${ALOHAMINI_DATASET_HOME:-$OPS_DIR/../datasets/lerobot}"
CALIBRATION_HOME="${ALOHAMINI_CALIBRATION_HOME:-$HOME/.cache/huggingface/lerobot/calibration}"
EVAL_LOG="${LOCAL_EVAL_LOG:-/tmp/alohamini_eval.log}"
mkdir -p "$DATASET_HOME"
printf '\n===== eval session started %s =====\n' "$(date '+%F %T')" >> "$EVAL_LOG"
exec > >(tee -a "$EVAL_LOG") 2>&1

source "$CONDA_INIT_LOCAL"
conda activate "$CONDA_ENV"
cd "$LOCAL_REPO"

echo "AlohaMini policy evaluation."
echo "Dataset root: $DATASET_HOME"
echo "Calibration root: $CALIBRATION_HOME"
echo "Full log: $EVAL_LOG"
echo "Robot model: $ROBOT_MODEL"
echo "Pi target: $PI_USER@$PI_HOST"

if [[ "${1:-}" == "--check_model" ]]; then
  shift
  MODEL_PATH="${1:?model path is required}"
  HF_LEROBOT_HOME="$DATASET_HOME" HF_LEROBOT_CALIBRATION="$CALIBRATION_HOME" \
    python -u examples/alohamini/check_policy_load.py --hf_model_id "$MODEL_PATH"
  exit 0
fi

HF_LEROBOT_HOME="$DATASET_HOME" HF_LEROBOT_CALIBRATION="$CALIBRATION_HOME" \
  ALOHAMINI_CAMERAS="${ALOHAMINI_CAMERAS:-forward,wrist_right}" \
  python -u examples/alohamini/evaluate_bi.py "$@"
