#!/usr/bin/env bash
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$OPS_DIR/config.env"
PI_USER="${ALOHAMINI_RUNTIME_PI_USER:-$PI_USER}"
PI_HOST="${ALOHAMINI_RUNTIME_PI_HOST:-$PI_HOST}"

if [[ ! -e /dev/am_arm_leader_left || ! -e /dev/am_arm_leader_right ]]; then
  echo "Leader serial links are missing: /dev/am_arm_leader_left or /dev/am_arm_leader_right"
  echo "Check USB connection and udev mapping before starting recording."
  exit 1
fi

if [[ ! -r /dev/am_arm_leader_left || ! -w /dev/am_arm_leader_left || ! -r /dev/am_arm_leader_right || ! -w /dev/am_arm_leader_right ]]; then
  echo "Leader serial links exist but are not readable/writable by this shell."
  echo "Current groups: $(id -nG)"
  echo "Expected group: dialout. Log out/in if dialout was added recently."
  exit 1
fi

DATASET_HOME="${ALOHAMINI_DATASET_HOME:-$OPS_DIR/../datasets/lerobot}"
CALIBRATION_HOME="${ALOHAMINI_CALIBRATION_HOME:-$HOME/.cache/huggingface/lerobot/calibration}"
RECORD_LOG="${LOCAL_RECORD_LOG:-/tmp/alohamini_record.log}"
mkdir -p "$DATASET_HOME"
printf '\n===== record session started %s =====\n' "$(date '+%F %T')" >> "$RECORD_LOG"
exec > >(tee -a "$RECORD_LOG") 2>&1

LEFT_CALIBRATION="$CALIBRATION_HOME/teleoperators/so_leader/${LEADER_ID}_left.json"
RIGHT_CALIBRATION="$CALIBRATION_HOME/teleoperators/so_leader/${LEADER_ID}_right.json"
if [[ ! -f "$LEFT_CALIBRATION" || ! -f "$RIGHT_CALIBRATION" ]]; then
  echo "Leader calibration files are missing."
  echo "Expected:"
  echo "  $LEFT_CALIBRATION"
  echo "  $RIGHT_CALIBRATION"
  echo "Run calibration first from the GUI Calibration page."
  exit 1
fi

source "$CONDA_INIT_LOCAL"
conda activate "$CONDA_ENV"
cd "$LOCAL_REPO"

echo "Starting AlohaMini dataset recording."
echo "Dataset root: $DATASET_HOME"
echo "Calibration root: $CALIBRATION_HOME"
echo "Full log: $RECORD_LOG"
echo "Robot model: $ROBOT_MODEL"
echo "Leader id: $LEADER_ID"
echo "Arm profile: $ARM_PROFILE"

python - <<'PY'
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

expected = {
    "/dev/am_arm_leader_left": range(1, 8),
    "/dev/am_arm_leader_right": range(1, 8),
}

failed = False
for port, ids in expected.items():
    motors = {
        f"motor_{mid}": Motor(mid, "sts3215", MotorNormMode.RANGE_0_100)
        for mid in ids
    }
    bus = FeetechMotorsBus(port=port, motors=motors)
    try:
        bus.connect()
        print(f"Leader preflight OK: {port}")
    except Exception as exc:
        failed = True
        print(f"Leader preflight FAILED: {port}")
        print(exc)
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass

if failed:
    raise SystemExit("Leader servo preflight failed. Fix Leader power/cabling/servo IDs before recording.")
PY

printf '\n\n' | HF_LEROBOT_HOME="$DATASET_HOME" HF_LEROBOT_CALIBRATION="$CALIBRATION_HOME" \
  ALOHAMINI_CAMERAS="${ALOHAMINI_CAMERAS:-forward,wrist_right}" \
  python -u examples/alohamini/record_bi.py \
  --remote_ip "$PI_HOST" \
  --robot_model "$ROBOT_MODEL" \
  --leader_id "$LEADER_ID" \
  --arm_profile "$ARM_PROFILE" \
  "$@"
