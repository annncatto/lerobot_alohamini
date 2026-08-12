#!/usr/bin/env python3
"""Headless load/control smoke test for the maintained alohamini_sim Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np


DATA_ENGINE = Path(__file__).resolve().parents[1] / "data_engine"
if str(DATA_ENGINE) not in sys.path:
    sys.path.insert(0, str(DATA_ENGINE))
import agents.aloha_mini  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="open the ManiSkill viewer")
    parser.add_argument("--steps", type=int, default=1, help="hold-current control steps")
    args = parser.parse_args()
    env = gym.make(
        "Empty-v1",
        robot_uids="alohamini2pro_sim",
        obs_mode="state",
        control_mode="pd_joint_pos_fixed_base",
        render_mode="human" if args.render else None,
        sim_backend="physx_cpu",
    )
    try:
        env.reset(seed=0)
        agent = env.unwrapped.agent
        qpos = agent.robot.get_qpos()[0].detach().cpu().numpy()
        names = [joint.name for joint in agent.robot.active_joints]
        by_name = dict(zip(names, qpos, strict=True))
        action = np.asarray(
            [0.0, 0.0, 0.0, by_name["vertical_move"]]
            + [by_name[name] for name in agent.left_arm_joint_names]
            + [by_name["left_gripper"]]
            + [by_name[name] for name in agent.right_arm_joint_names]
            + [by_name["right_gripper"]],
            dtype=np.float32,
        )
        for _ in range(max(1, args.steps)):
            env.step(action)
            if args.render:
                env.render()
        report = {
            "passed": True,
            "qpos_shape": list(agent.robot.get_qpos().shape),
            "action_shape": list(env.action_space.shape),
            "link_count": len(agent.robot.get_links()),
            "runtime_mass_kg": sum(float(link.get_mass()) for link in agent.robot.get_links()),
            "left_tcp_m": agent.tcp_pos[0].detach().cpu().tolist(),
        }
        if report["qpos_shape"] != [1, 18] or report["action_shape"] != [18]:
            raise RuntimeError(f"unexpected robot dimensions: {report}")
        if not np.isclose(report["runtime_mass_kg"], 16.0, atol=1e-4):
            raise RuntimeError(f"unexpected runtime mass: {report}")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
