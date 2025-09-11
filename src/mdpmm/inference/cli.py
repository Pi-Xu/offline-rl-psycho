from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from mdpmm.envs import make_env
from mdpmm.models.dqn import DQNAgent, ReplayBuffer, Transition
from mdpmm.utils.config import AppSettings, TrainDqnConfig, load_train_config, materialize_run_dir
from mdpmm.utils.io import append_jsonl, ensure_dir, save_json
from mdpmm.utils.logging import setup_logging
from mdpmm.utils.seeding import set_global_seeds


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    f = max(0.0, min(1.0, 1.0 - step / float(decay_steps)))
    return end + (start - end) * f


def evaluate(env_id: str, agent: DQNAgent, episodes: int, max_steps: int, seed: int) -> Dict[str, float]:
    env = make_env(env_id)
    rng = np.random.RandomState(seed + 123)
    returns = []
    steps_list = []
    solved_count = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        total_r = 0.0
        steps = 0
        for t in range(max_steps):
            legal = info["action_mask"]
            action = agent.act(obs, legal_mask=legal, epsilon=0.0)
            sr = env.step(action)
            obs, info = sr.obs, {"action_mask": env.legal_action_mask()}
            total_r += sr.reward
            steps += 1
            if sr.done:
                if sr.info.get("solved"):
                    solved_count += 1
                break
        returns.append(total_r)
        steps_list.append(steps)
    success_rate = float(solved_count / episodes)
    return {
        "success_rate": success_rate,
        "avg_return": float(np.mean(returns) if returns else 0.0),
        "avg_steps": float(np.mean(steps_list) if steps_list else 0.0),
    }


def train_dqn(config: TrainDqnConfig) -> None:
    logger = setup_logging(name="mdpmm.train")
    AppSettings()  # load .env defaults if present
    seed = set_global_seeds(config.seed)

    run_dir = materialize_run_dir(config.artifacts_dir, config.run_id)
    ensure_dir(run_dir)
    save_json(json.loads(config.model_dump_json()), os.path.join(run_dir, "config.json"))

    env = make_env(config.env_id)
    obs_dim = int(np.prod(env.obs_shape))
    num_actions = env.num_actions
    agent = DQNAgent(obs_dim, num_actions, lr=config.lr, gamma=config.gamma, device=config.device)
    buf = ReplayBuffer(config.replay_capacity)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best = {"success_rate": -1.0, "avg_steps": float("inf")}
    global_step = 0
    eval_stats = {}

    for ep in range(1, config.train_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        total_r = 0.0
        steps = 0
        for t in range(config.max_steps_per_episode):
            epsilon = linear_epsilon(global_step, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
            legal = info["action_mask"]
            action = agent.act(obs, legal_mask=legal, epsilon=epsilon)
            sr = env.step(action)
            next_obs = sr.obs
            done = sr.done
            next_legal = env.legal_action_mask()
            buf.add(
                Transition(
                    s=obs,
                    a=action,
                    r=sr.reward,
                    ns=next_obs,
                    done=done,
                    legal_mask_s=legal,
                    legal_mask_ns=next_legal,
                )
            )
            obs, info = next_obs, {"action_mask": next_legal}
            total_r += sr.reward
            steps += 1

            if len(buf) >= config.learning_starts and (global_step % 1 == 0):
                batch = buf.sample(config.batch_size)
                stats = agent.update(batch)
                if global_step % 100 == 0:
                    append_jsonl({"step": global_step, "kind": "train", **stats}, metrics_path)

            if global_step % config.target_update_interval == 0 and global_step > 0:
                agent.sync_target()

            global_step += 1
            if done:
                break

        # Per-episode log
        append_jsonl(
            {
                "episode": ep,
                "kind": "episode",
                "return": total_r,
                "steps": steps,
                "epsilon": epsilon,
            },
            metrics_path,
        )

        # Periodic evaluation
        if ep % config.eval_every == 0 or ep == config.train_episodes:
            eval_stats = evaluate(
                env_id=config.env_id,
                agent=agent,
                episodes=config.eval_episodes,
                max_steps=config.max_steps_per_episode,
                seed=seed + 999,
            )
            append_jsonl({"episode": ep, "kind": "eval", **eval_stats}, metrics_path)

            # Save last checkpoint
            last_path = os.path.join(run_dir, "last.pt")
            agent.save(last_path, meta={"episode": ep, **eval_stats})

            # Track best (by success_rate then shorter avg_steps)
            is_better = (eval_stats["success_rate"] > best["success_rate"]) or (
                eval_stats["success_rate"] == best["success_rate"] and eval_stats["avg_steps"] < best["avg_steps"]
            )
            if is_better:
                best = {"success_rate": eval_stats["success_rate"], "avg_steps": eval_stats["avg_steps"]}
                best_path = os.path.join(run_dir, "best.pt")
                agent.save(best_path, meta={"episode": ep, **eval_stats})

    # Write final summary
    save_json({"best": best, "last_eval": eval_stats}, os.path.join(run_dir, "summary.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="MDPMM CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train-dqn", help="Train DQN on a registered env")
    p_train.add_argument("--config", type=str, default=None, help="Path to YAML config")

    args = parser.parse_args()

    if args.cmd == "train-dqn":
        cfg = load_train_config(args.config)
        train_dqn(cfg)


if __name__ == "__main__":
    main()

