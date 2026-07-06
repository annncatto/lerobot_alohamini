#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch

import lerobot.robots.alohamini  # noqa: F401

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.utils.device_utils import auto_select_torch_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a pretrained LeRobot policy can be loaded.")
    parser.add_argument("--hf_model_id", required=True, help="Local checkpoint directory or Hub model id.")
    args = parser.parse_args()

    model_path = Path(args.hf_model_id)
    if model_path.exists() and not (model_path / "config.json").exists():
        raise SystemExit(f"Missing config.json in model directory: {model_path}")

    device = str(auto_select_torch_device())
    policy_cfg = PreTrainedConfig.from_pretrained(args.hf_model_id)
    policy_cfg.pretrained_path = args.hf_model_id
    policy = get_policy_class(policy_cfg.type).from_pretrained(args.hf_model_id, config=policy_cfg)
    policy = policy.to(device)
    policy.eval()

    total_params = sum(param.numel() for param in policy.parameters())
    trainable_params = sum(param.numel() for param in policy.parameters() if param.requires_grad)
    print(f"Policy type: {policy_cfg.type}")
    print(f"Device: {device}")
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print("Policy load check OK.")


if __name__ == "__main__":
    main()
