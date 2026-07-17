#!/usr/bin/env bash
set -euo pipefail

REPO=/home/anncatto/lerobot_alohamini
PYTHON=/home/anncatto/miniconda3/envs/lerobot_alohamini/bin/python
DATA=/home/anncatto/alohamini_gui/datasets/lerobot/local/alohamini_04_visual_only_no_chest
RUN=${RUN:-act_alohamini04_visual_only_forward_wrist_vae_200k_$(date +%Y%m%d_%H%M%S)}

export HF_HOME=/home/anncatto/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
exec "$PYTHON" -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/alohamini_04_visual_only_no_chest \
  --dataset.root="$DATA" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.use_vae=true \
  --policy.fixed_action_dims='[0,1,2,3,4,5,6]' \
  --policy.push_to_hub=false \
  --policy.repo_id="local/$RUN" \
  --output_dir="$REPO/outputs/$RUN" \
  --job_name="$RUN" \
  --wandb.enable=false \
  --steps=200000 \
  --batch_size=2 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=10000
