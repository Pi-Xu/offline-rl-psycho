from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from mdpmm.envs import make_env
from mdpmm.models.dqn import DQNAgent
from mdpmm.utils.config import GenerateDataConfig
from mdpmm.utils.io import ensure_dir, append_jsonl, save_json
from mdpmm.utils.seeding import set_global_seeds


def _masked_softmax(q: np.ndarray, mask: np.ndarray, beta: float) -> np.ndarray:
    """Compute softmax over legal actions only: pi(a|s, beta) ∝ exp(beta * Q(s,a)).

    Args:
        q: shape [A], action-values
        mask: shape [A] boolean mask for legal actions
        beta: inverse temperature (>0)
    Returns:
        probs over A with zeros on illegal actions and sum=1 over legal.
    """
    assert q.ndim == 1 and mask.ndim == 1 and q.shape[0] == mask.shape[0]
    legal_idx = np.flatnonzero(mask)
    if len(legal_idx) == 0:
        # No legal moves; return uniform zeros (caller should terminate)
        p = np.zeros_like(q, dtype=np.float64)
        return p
    x = beta * q[legal_idx]
    # numerical stability: subtract max
    x = x - np.max(x)
    ex = np.exp(x)
    p_legal = ex / np.sum(ex)
    p = np.zeros_like(q, dtype=np.float64)
    p[legal_idx] = p_legal
    return p


def _sample_action(probs: np.ndarray, rng: np.random.RandomState) -> int:
    a = int(rng.choice(len(probs), p=probs))
    return a


@dataclass
class RolloutRecord:
    pid: int
    episode: int
    t: int
    beta: float
    s: List[float]
    a: int
    r: float
    ns: List[float]
    done: bool
    legal_s: List[bool]
    legal_ns: List[bool]


def generate_dataset(cfg: GenerateDataConfig) -> str:
    """Generate offline trajectories from a trained checkpoint using a softmax policy.

    Returns the output directory path where files are written.
    """
    seed = set_global_seeds(cfg.seed)
    rng = np.random.RandomState(seed + 2025)

    ckpt_path = cfg.resolve_checkpoint()
    env = make_env(cfg.env_id)
    obs_dim = int(np.prod(env.obs_shape))
    agent, meta = DQNAgent.from_checkpoint(
        ckpt_path,
        obs_dim=obs_dim,
        num_actions=env.num_actions,
        lr=3e-4,  # not used
        gamma=0.99,  # not used
        device=cfg.device,
        model_type=cfg.model_type,
        obs_shape=env.valid_mask.shape,
        cnn_channels=tuple(cfg.cnn_channels),
        cnn_hidden=int(cfg.cnn_hidden),
    )

    # Prepare output folder
    from datetime import datetime

    run_name = cfg.out_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(cfg.out_dir, run_name))
    traj_path = os.path.join(out_dir, "trajectories.jsonl")

    # Pre-sample participant betas
    if cfg.beta_mode.lower() == "lognormal":
        z = rng.normal(loc=cfg.beta_mu, scale=cfg.beta_sigma, size=cfg.participants)
        betas = np.exp(z)
    elif cfg.beta_mode.lower() == "fixed":
        betas = np.full(cfg.participants, float(cfg.beta_fixed))
    else:
        raise ValueError("beta_mode must be one of {lognormal, fixed}")

    # Write a small manifest
    manifest = {
        "env_id": cfg.env_id,
        "seed": cfg.seed,
        "participants": cfg.participants,
        "episodes_per_participant": cfg.episodes_per_participant,
        "max_steps_per_episode": cfg.max_steps_per_episode,
        "beta_mode": cfg.beta_mode,
        "beta_mu": cfg.beta_mu,
        "beta_sigma": cfg.beta_sigma,
        "beta_fixed": cfg.beta_fixed,
        "checkpoint": os.path.abspath(ckpt_path),
        "model_type": cfg.model_type,
        "cnn_channels": list(cfg.cnn_channels),
        "cnn_hidden": int(cfg.cnn_hidden),
        "meta": meta,
    }
    save_json(manifest, os.path.join(out_dir, "manifest.json"))

    # Rollouts
    for pid in range(cfg.participants):
        beta = float(betas[pid])
        for ep in range(cfg.episodes_per_participant):
            # stochastic reset seed per episode for diversity (but deterministic overall)
            obs, info = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            for t in range(cfg.max_steps_per_episode):
                # Compute Q values
                with torch.no_grad():
                    q = agent.q(torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0))
                    q_np = q.squeeze(0).detach().cpu().numpy().astype(np.float64)
                mask = info["action_mask"].astype(bool)
                probs = _masked_softmax(q_np, mask, beta)
                if probs.sum() <= 0:
                    # No legal move – terminate
                    break
                a = _sample_action(probs, rng)
                sr = env.step(a)
                rec = RolloutRecord(
                    pid=pid,
                    episode=ep,
                    t=t,
                    beta=beta,
                    s=obs.astype(float).tolist(),
                    a=int(a),
                    r=float(sr.reward),
                    ns=sr.obs.astype(float).tolist(),
                    done=bool(sr.done),
                    legal_s=mask.astype(bool).tolist(),
                    legal_ns=env.legal_action_mask().astype(bool).tolist(),
                )
                append_jsonl(json.loads(json.dumps(rec.__dict__)), traj_path)
                obs, info = sr.obs, {"action_mask": env.legal_action_mask()}
                if sr.done:
                    break

    return out_dir


def main() -> None:  # pragma: no cover - thin CLI
    import argparse

    p = argparse.ArgumentParser(description="Generate offline trajectories from a best.pt using softmax policy")
    p.add_argument("--env-id", default="peg7x7")
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--participants", type=int, default=50)
    p.add_argument("--episodes-per-participant", type=int, default=2)
    p.add_argument("--max-steps-per-episode", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta-mode", choices=["lognormal", "fixed"], default="lognormal")
    p.add_argument("--beta-mu", type=float, default=0.0)
    p.add_argument("--beta-sigma", type=float, default=0.5)
    p.add_argument("--beta-fixed", type=float, default=1.0)
    p.add_argument("--out-dir", default="artifacts/datasets/synth")
    p.add_argument("--out-name", default=None)
    p.add_argument("--model-type", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--cnn-channels", type=str, default="[16,32]")
    p.add_argument("--cnn-hidden", type=int, default=256)

    args = p.parse_args()
    # lightweight parsing for channels
    def _parse_channels(s: str) -> Tuple[int, int]:
        try:
            xs = json.loads(s)
            if isinstance(xs, list) and len(xs) >= 2:
                return int(xs[0]), int(xs[1])
        except Exception:
            pass
        return (16, 32)

    cfg = GenerateDataConfig(
        env_id=args.env_id,
        checkpoint_path=args.checkpoint_path,
        run_dir=args.run_dir,
        device=args.device,
        participants=args.participants,
        episodes_per_participant=args.episodes_per_participant,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        beta_mode=args.beta_mode,
        beta_mu=args.beta_mu,
        beta_sigma=args.beta_sigma,
        beta_fixed=args.beta_fixed,
        out_dir=args.out_dir,
        out_name=args.out_name,
        model_type=args.model_type,
        cnn_channels=_parse_channels(args.cnn_channels),
        cnn_hidden=args.cnn_hidden,
    )
    out_dir = generate_dataset(cfg)
    print(out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()

