#!/usr/bin/env bash
set -euo pipefail

REPO=/home/anncatto/lerobot_alohamini
PYTHON=/home/anncatto/miniconda3/envs/lerobot_alohamini/bin/python
DATA=/home/anncatto/alohamini_gui/datasets/lerobot/local/alohamini_01
RUN=act_alohamini01_local_20260714_2040

export HF_HOME=/home/anncatto/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
exec "$PYTHON" -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/alohamini_01 \
  --dataset.root="$DATA" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.push_to_hub=false \
  --policy.repo_id="local/$RUN" \
  --output_dir="$REPO/outputs/$RUN" \
  --job_name="$RUN" \
  --wandb.enable=false \
  --steps=100000 \
  --batch_size=2 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=10000
