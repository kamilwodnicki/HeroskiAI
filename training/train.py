"""
Minimal, single-environment PPO training loop for the VCMI battle model.

Reuses from vcmi-gym's own rl/v15/gnn_model.py (NOT reimplemented here, see README):
  - GNNModel              the graph neural network (policy + value heads)
  - to_hdata              converts one raw env observation into a PyG HeteroData graph
  - add_action_active_local_ids   computes batched indices for the legal-action mask

Everything else (the rollout loop, GAE, the PPO update, checkpointing) is custom,
intentionally simpler than vcmi-gym's own rl/v15/ppo_gnn.py (no W&B, no multi-env
vectorization, no distributed/eval machinery) so the mechanics stay visible.
"""

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

sys.path.insert(0, "/vcmi-gym")

from vcmi_gym import VcmiEnv_v15  # noqa: E402
from rl.v15.gnn_model import GNNModel, to_hdata, add_action_active_local_ids  # noqa: E402

DEVICE = torch.device("cpu")  # this image only installs the CPU build of torch

# These two edge types dominate the graph (tens of thousands of edges) and are
# dropped by vcmi-gym's own default training config for that reason.
IGNORED_EDGES = [
    ["Hex", "BecomesMeleeTargetAfter", "Action"],
    ["Hex", "BecomesShootTargetAfter", "Action"],
]

ENV_KWARGS = dict(
    mapname="gym/A1.vmap",  # small, known-good map used by vcmi-gym itself for quick tests
    opponent="StupidAI",
    max_steps=500,
    ignored_edges=IGNORED_EDGES,
    vcmi_loglevel_global="error",
    vcmi_loglevel_ai="error",
    vcmienv_loglevel="WARN",
    # VcmiEnv's own defaults for these three are 1 (full-scale 0-1000 "permille"
    # army-value deltas), which blows up episode returns into the thousands.
    # vcmi-gym's own tuned training config uses 0.01 to keep returns small and
    # the value function well-conditioned -- reuse that here.
    reward_dmg_mult=0.01,
    reward_term_mult=0.01,
    reward_relval_mult=0.01,
    # Previously both armies were fixed every episode, so the model could win
    # by memorizing one exact battle instead of learning general combat skill.
    # Randomize army composition/strength each episode (same ranges vcmi-gym's
    # own training config uses) so it actually has to generalize.
    random_armies=True,
    random_army_value_min=5_000,
    random_army_value_max=50_000,
    random_army_target_var=30,
)

MODEL_CONFIG = dict(
    gnn_num_layers=2,
    gnn_hidden_channels=64,
    gnn_out_channels=32,
    gnn_conv_cls="GENConv",
    gnn_conv_kwargs=dict(aggr="softmax", learn_t=True, num_layers=2, norm=None),
    policy_head_hidden_channels=64,
    value_head_hidden_channels=64,
)

TRAIN_CONFIG = dict(
    num_steps=128,          # rollout length collected before each PPO update
    num_minibatches=4,
    update_epochs=4,
    learning_rate=1e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=1.0,
    total_iterations=10_000,
    log_every=5,
    checkpoint_every=50,
    checkpoint_dir="/vcmi-gym/checkpoints",
)


def reset_env(env):
    obs, info = env.reset()
    # env.reset() skips the edge filtering that env.step() applies -- apply it
    # ourselves so every obs we hand to to_hdata() has a consistent edge schema.
    return env.filter_obs(obs), info


def build_batch(obs_list, done_list):
    hdatas = [to_hdata(obs, done=torch.tensor(bool(d))) for obs, d in zip(obs_list, done_list)]
    batch = Batch.from_data_list(hdatas)
    add_action_active_local_ids(batch)
    return batch


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = VcmiEnv_v15(**ENV_KWARGS)

    node_types = VcmiEnv_v15.node_types()
    edge_types = VcmiEnv_v15.filtered_edge_types(IGNORED_EDGES)
    model = GNNModel(node_types, edge_types, MODEL_CONFIG).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])

    os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)

    obs, _ = reset_env(env)

    finished_returns = []
    finished_wins = []
    running_ep_return = 0.0

    num_steps = TRAIN_CONFIG["num_steps"]

    for iteration in range(1, TRAIN_CONFIG["total_iterations"] + 1):
        b_obs = []
        b_actions = np.zeros(num_steps, dtype=np.int64)
        b_logprobs = np.zeros(num_steps, dtype=np.float32)
        b_values = np.zeros(num_steps, dtype=np.float32)
        b_rewards = np.zeros(num_steps, dtype=np.float32)
        b_dones = np.zeros(num_steps, dtype=np.float32)

        # --- 1. collect a rollout of num_steps transitions ---
        model.eval()
        with torch.no_grad():
            for t in range(num_steps):
                hdata = build_batch([obs], [False])
                value = model.forward_value(hdata)
                action, logprob, _ = model.forward_policy(hdata, deterministic=False)

                action_id = int(action.item())
                next_obs, reward, term, trunc, info = env.step(action_id)
                done = term or trunc

                b_obs.append(obs)
                b_actions[t] = action_id
                b_logprobs[t] = logprob.item()
                b_values[t] = value.item()
                b_rewards[t] = reward
                b_dones[t] = float(done)

                running_ep_return += reward

                if done:
                    finished_returns.append(running_ep_return)
                    finished_wins.append(bool(info.get("is_success", False)))
                    running_ep_return = 0.0
                    obs, _ = reset_env(env)
                else:
                    obs = next_obs

            # bootstrap value for whatever comes after the last stored step
            next_value = model.forward_value(build_batch([obs], [False])).item()

        # --- 2. GAE: advantage/return per step ---
        advantages = np.zeros(num_steps, dtype=np.float32)
        lastgae = 0.0
        for t in reversed(range(num_steps)):
            nextnonterminal = 1.0 - b_dones[t]
            nextvalue = b_values[t + 1] if t + 1 < num_steps else next_value
            delta = b_rewards[t] + TRAIN_CONFIG["gamma"] * nextvalue * nextnonterminal - b_values[t]
            lastgae = delta + TRAIN_CONFIG["gamma"] * TRAIN_CONFIG["gae_lambda"] * nextnonterminal * lastgae
            advantages[t] = lastgae
        returns = advantages + b_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_actions = torch.as_tensor(b_actions, device=DEVICE)
        t_logprobs = torch.as_tensor(b_logprobs, device=DEVICE)
        t_advantages = torch.as_tensor(advantages, device=DEVICE)
        t_returns = torch.as_tensor(returns, device=DEVICE)
        t_dones = b_dones

        # --- 3. PPO update: several epochs over shuffled minibatches ---
        model.train()
        minibatch_size = num_steps // TRAIN_CONFIG["num_minibatches"]
        indices = np.arange(num_steps)

        for _ in range(TRAIN_CONFIG["update_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, num_steps, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                mb_obs = [b_obs[i] for i in mb_idx]
                mb_dones = [t_dones[i] for i in mb_idx]
                hdata = build_batch(mb_obs, mb_dones)

                mb_action = t_actions[mb_idx]
                _, new_logprob, entropy = model.forward_policy(hdata, b_action=mb_action)
                new_value = model.forward_value(hdata).squeeze(-1)

                logratio = new_logprob - t_logprobs[mb_idx]
                ratio = logratio.exp()

                mb_adv = t_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - TRAIN_CONFIG["clip_coef"], 1 + TRAIN_CONFIG["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (new_value - t_returns[mb_idx]).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - TRAIN_CONFIG["ent_coef"] * entropy_loss + TRAIN_CONFIG["vf_coef"] * v_loss
                last_pg_loss, last_v_loss, last_entropy = pg_loss.item(), v_loss.item(), entropy_loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["max_grad_norm"])
                optimizer.step()

        # --- 4. logging + checkpointing ---
        if iteration % TRAIN_CONFIG["log_every"] == 0:
            mean_return = np.mean(finished_returns[-50:]) if finished_returns else float("nan")
            win_rate = np.mean(finished_wins[-50:]) if finished_wins else float("nan")
            print(
                f"iter={iteration} episodes={len(finished_returns)} "
                f"mean_return(last50)={mean_return:.2f} win_rate(last50)={win_rate:.2f} "
                f"pg_loss={last_pg_loss:.3f} v_loss={last_v_loss:.3f} entropy={last_entropy:.3f}",
                flush=True,
            )

        if iteration % TRAIN_CONFIG["checkpoint_every"] == 0:
            ckpt_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "latest.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
            }, ckpt_path)
            print(f"checkpoint saved: {ckpt_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
