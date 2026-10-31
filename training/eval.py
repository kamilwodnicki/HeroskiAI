"""
Evaluate a saved checkpoint with a deterministic (greedy) policy -- no training,
no exploration. Reuses train.py's env/model setup so the model built here always
matches what was saved in the checkpoint.

Usage:
    .venv/bin/python training/eval.py --opponent BattleAI --episodes 50
"""

import argparse

import numpy as np
import torch

from train import (
    DEVICE,
    ENV_KWARGS,
    MODEL_CONFIG,
    IGNORED_EDGES,
    VcmiEnv_v15,
    GNNModel,
    build_batch,
    reset_env,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/vcmi-gym/checkpoints/latest.pt")
    parser.add_argument("--opponent", default="BattleAI")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    env = VcmiEnv_v15(**dict(ENV_KWARGS, opponent=args.opponent))

    node_types = VcmiEnv_v15.node_types()
    edge_types = VcmiEnv_v15.filtered_edge_types(IGNORED_EDGES)
    model = GNNModel(node_types, edge_types, MODEL_CONFIG).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from iteration {ckpt['iteration']}, evaluating vs {args.opponent}")

    wins = []
    returns = []
    obs, _ = reset_env(env)
    ep_return = 0.0

    with torch.no_grad():
        while len(wins) < args.episodes:
            hdata = build_batch([obs], [False])
            action, _, _ = model.forward_policy(hdata, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action.item()))
            ep_return += reward

            if term or trunc:
                wins.append(bool(info.get("is_success", False)))
                returns.append(ep_return)
                ep_return = 0.0
                obs, _ = reset_env(env)

    print(f"episodes={len(wins)} win_rate={np.mean(wins):.2f} mean_return={np.mean(returns):.2f}")


if __name__ == "__main__":
    main()
