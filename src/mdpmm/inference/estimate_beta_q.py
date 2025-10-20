from __future__ import annotations

"""
Beta–Q estimation from interaction logs (trajectories.jsonl).

Implements a light-weight version of the generalized EM loop described in
doc/25_08_23_rl4psycho.md, focusing on:
 - E-step: per-participant beta update via 1D Newton steps on z=log beta
 - M-step: shared Q(s,a) update by minimizing behavior NLL with optional
           soft Bellman and CQL regularizers using observed rewards.

This module is intentionally simple and CPU-friendly for small offline runs.
"""

import json
import math
import os
from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mdpmm.utils.seeding import set_global_seeds
from mdpmm.utils.logging import setup_logging


# -------------------------------
# Data IO
# -------------------------------


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@dataclass
class TrajBatch:
    s: torch.Tensor  # [N, S]
    a: torch.Tensor  # [N]
    r: torch.Tensor  # [N]
    ns: torch.Tensor  # [N, S]
    done: torch.Tensor  # [N]
    legal_s: torch.Tensor  # [N, A] bool
    legal_ns: torch.Tensor  # [N, A] bool
    pid: torch.Tensor  # [N]


def load_dataset(dataset_dir: str, device: str = "cpu") -> Tuple[TrajBatch, int, int, Dict[int, List[int]]]:
    """Load trajectories.jsonl into a flat batch and provide indexing by pid.

    Returns: (batch, state_dim, num_actions, pid_index)
    - pid_index maps pid -> list of row indices (for E-step updates)
    """
    traj_path = os.path.join(dataset_dir, "trajectories.jsonl")
    rows: List[Dict] = list(_iter_jsonl(traj_path))
    if not rows:
        raise ValueError(f"Empty trajectories at {traj_path}")

    # Infer shapes
    s_dim = len(rows[0]["s"])  # flattened state
    a_dim = len(rows[0]["legal_s"])  # number of discrete actions

    # Build arrays
    s = torch.tensor([r["s"] for r in rows], dtype=torch.float32, device=device)
    a = torch.tensor([r["a"] for r in rows], dtype=torch.long, device=device)
    rwd = torch.tensor([r["r"] for r in rows], dtype=torch.float32, device=device)
    ns = torch.tensor([r["ns"] for r in rows], dtype=torch.float32, device=device)
    done = torch.tensor([1.0 if r["done"] else 0.0 for r in rows], dtype=torch.float32, device=device)
    legal_s = torch.tensor([r["legal_s"] for r in rows], dtype=torch.bool, device=device)
    legal_ns = torch.tensor([r["legal_ns"] for r in rows], dtype=torch.bool, device=device)
    pid = torch.tensor([r["pid"] for r in rows], dtype=torch.long, device=device)

    # Index by pid
    pid_index: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        pid_index.setdefault(int(r["pid"]), []).append(i)

    batch = TrajBatch(s=s, a=a, r=rwd, ns=ns, done=done, legal_s=legal_s, legal_ns=legal_ns, pid=pid)
    return batch, s_dim, a_dim, pid_index


# -------------------------------
# Models and helpers
# -------------------------------


class MLPQ(nn.Module):
    """Simple MLP that maps state -> Q(s, a) over all actions.

    This differs from mdpmm.models.dqn.MLPQ by keeping the dependency
    surface minimal and not requiring env metadata.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # [N,S] -> [N,A]
        return self.net(s)


def masked_logsumexp(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute logsumexp over entries where mask is True.

    Args:
        x: [*, A]
        mask: [*, A] boolean
    Returns: [*] logsumexp(x[mask]) stable, with -inf if no True entry.
    """
    # Replace masked-out entries by -inf so they don't contribute
    neg_inf = torch.tensor(-1e9, dtype=x.dtype, device=x.device)
    x_masked = torch.where(mask, x, neg_inf)
    m, _ = torch.max(x_masked, dim=dim, keepdim=True)
    z = torch.sum(torch.where(mask, torch.exp(x_masked - m), torch.zeros_like(x)), dim=dim, keepdim=False)
    return (m.squeeze(dim) + torch.log(torch.clamp(z, min=1e-12))).to(x.dtype)


def masked_softmax(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    lse = masked_logsumexp(x, mask, dim=dim).unsqueeze(dim)
    ex = torch.exp(x - lse)
    return torch.where(mask, ex, torch.zeros_like(ex))


# -------------------------------
# E-step: beta updates per participant
# -------------------------------


@dataclass
class BetaPrior:
    mu: float = 0.0  # prior on z=log beta
    sigma: float = 0.5


def advantage_from_q(
    q: torch.Tensor,
    legal: torch.Tensor,
    *,
    normalize: str = "meanstd",
    eps: float = 1e-6,
    clamp: Optional[float] = None,
) -> torch.Tensor:
    """Convert Q logits to an advantage form over legal actions.

    Args:
        q: [N, A] unnormalized action values.
        legal: [N, A] bool mask indicating available actions.
        normalize: one of {"none", "mean", "meanstd"}.
            - "none": return masked Q (no centering).
            - "mean": subtract per-state mean over legal actions.
            - "meanstd": subtract mean and divide by per-state std (clamped by eps).
        eps: lower bound applied to the per-state standard deviation.
        clamp: if provided, clip normalized logits into [-clamp, clamp].

    Returns:
        Tensor of logits with illegal actions zeroed out so downstream masked
        softmax ignores them.
    """
    if normalize not in {"none", "mean", "meanstd"}:
        raise ValueError(f"normalize must be one of 'none', 'mean', 'meanstd'; got {normalize}")

    mask = legal.to(dtype=q.dtype)
    masked_q = torch.where(legal, q, torch.zeros_like(q))
    counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    if normalize == "none":
        return masked_q

    mean = (masked_q.sum(dim=-1, keepdim=True) / counts)
    diff = torch.where(legal, q - mean, torch.zeros_like(q))
    if normalize == "mean":
        out = diff
        return torch.clamp(out, min=-clamp, max=clamp) if clamp is not None else out

    var = ((diff**2) * mask).sum(dim=-1, keepdim=True) / counts
    # Clamp variance before sqrt to keep gradients finite when only one action is legal.
    var = torch.clamp(var, min=eps**2)
    std = torch.sqrt(var)
    scale = torch.maximum(std, torch.full_like(std, eps))
    out = diff / scale
    return torch.clamp(out, min=-clamp, max=clamp) if clamp is not None else out


def _e_step_newton(
    logits_all: torch.Tensor,  # [N, A]
    a: torch.Tensor,  # [N]
    mask: torch.Tensor,  # [N, A]
    indices: List[int],
    prior: BetaPrior,
    z_init: float | None = None,
    max_iter: int = 10,
    tol: float = 1e-6,
    *,
    line_search: bool = True,
    z_min: float = math.log(1e-3),
    z_max: float = math.log(1e3),
) -> float:
    """Maximize participant j's log posterior over z=log beta using Newton steps.

    Implements Eq. (5.1) gradients/Hessian from the design doc.
    Returns beta_j = exp(z_hat).
    """
    if len(indices) == 0:
        return float(math.exp(prior.mu))

    # Slice data for this participant
    phi = logits_all[indices]  # [T, A]
    aj = a[indices]  # [T]
    mj = mask[indices]  # [T, A]

    z = float(prior.mu if z_init is None else z_init)

    def _obj(zv: float) -> float:
        beta_ = math.exp(zv)
        logits_ = beta_ * phi
        lse_ = masked_logsumexp(logits_, mj, dim=-1)
        chosen_ = logits_.gather(1, aj.view(-1, 1)).squeeze(1)
        lp_ = torch.sum(chosen_ - lse_).item()
        prior_term_ = -0.5 * ((zv - prior.mu) ** 2) / (prior.sigma**2)
        return float(lp_ + prior_term_)

    for _ in range(max_iter):
        beta = math.exp(z)
        # pi_beta over legal actions
        logits = beta * phi  # [T, A]
        pi = masked_softmax(logits, mj, dim=-1)  # [T, A]
        # E_pi[phi] and Var_pi[phi]
        e_q = torch.sum(pi * phi, dim=-1)  # [T]
        e_q2 = torch.sum(pi * (phi**2), dim=-1)
        var_q = torch.clamp(e_q2 - e_q**2, min=0.0)
        # Gather chosen Q(s,a)
        q_chosen = phi.gather(1, aj.view(-1, 1)).squeeze(1)  # [T]

        # g and H (on log scale z)
        g = beta * torch.sum(q_chosen - e_q).item() - (z - prior.mu) / (prior.sigma**2)
        # The design doc includes an additional term; for stability, we use the dominant negative-curvature term + prior.
        H = - (beta**2) * torch.sum(var_q).item() - 1.0 / (prior.sigma**2)

        if not math.isfinite(g) or not math.isfinite(H) or H >= 0.0:
            # Fallback to small step if curvature is degenerate; keep concave update.
            H = -max(1.0, abs(g))

        step = g / H
        z_prop = z - step
        # Clamp to reasonable range to avoid extreme betas
        z_prop = max(z_min, min(z_max, z_prop))

        if not line_search:
            if abs(z_prop - z) < tol:
                z = z_prop
                break
            z = z_prop
            continue

        # Backtracking line search to guarantee non-decreasing objective
        obj0 = _obj(z)
        z_new = z_prop
        tries = 0
        while tries < 10:
            obj_new = _obj(z_new)
            if math.isfinite(obj_new) and obj_new >= obj0 - 1e-10:
                break
            # shrink towards current z
            z_new = z + 0.5 * (z_new - z)
            tries += 1
        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new

    return float(math.exp(z))


def update_betas_for_all(
    logits_all: torch.Tensor,
    a: torch.Tensor,
    mask: torch.Tensor,
    pid_index: Dict[int, List[int]],
    prior: BetaPrior,
    z0: float | None = None,
) -> Dict[int, float]:
    betas: Dict[int, float] = {}
    for pid, idxs in pid_index.items():
        betas[pid] = _e_step_newton(logits_all, a, mask, idxs, prior=prior, z_init=z0)
    return betas


# -------------------------------
# M-step: optimize Q given betas
# -------------------------------


@dataclass
class EstimatorConfig:
    dataset_dir: str
    out_dir: str = "artifacts/estimates"
    seed: int = 42
    device: str = "cpu"
    # Optimization
    lr: float = 1e-3
    batch_size: int = 256
    rounds: int = 3
    steps_per_round: int = 200
    # RL terms
    gamma: float = 0.99
    tau: float = 1.0  # soft Bellman temperature
    lambda_bell: float = 0.5
    lambda_cql: float = 0.1
    # Beta prior
    beta_mu: float = 0.0
    beta_sigma: float = 0.5
    # Model
    hidden: int = 128
    # Debug / numerical options
    debug: bool = False
    line_search: bool = True
    max_newton_iter: int = 10
    newton_tol: float = 1e-6
    beta_min: float = 1e-3
    beta_max: float = 1e3
    # Policy head
    policy_head: str = "q"  # {"q", "adv"}
    adv_normalize: str = "meanstd"
    adv_eps: float = 0.05
    adv_clip: Optional[float] = 20.0


def _make_policy_logits(
    q: torch.Tensor,
    legal: torch.Tensor,
    *,
    head: str,
    normalize: str,
    eps: float,
    clamp: Optional[float],
) -> torch.Tensor:
    if head == "q":
        return q
    if head == "adv":
        return advantage_from_q(q, legal, normalize=normalize, eps=eps, clamp=clamp)
    raise ValueError(f"Unknown policy_head '{head}'. Expected 'q' or 'adv'.")


def behavior_nll(logits: torch.Tensor, a: torch.Tensor, legal: torch.Tensor, beta_per_row: torch.Tensor) -> torch.Tensor:
    """-sum log pi(a|s,beta) over rows.

    q: [N, A], a: [N], legal: [N, A] bool, beta_per_row: [N]
    """
    scaled = logits * beta_per_row.view(-1, 1)
    lse = masked_logsumexp(scaled, legal, dim=-1)
    chosen = scaled.gather(1, a.view(-1, 1)).squeeze(1)
    return torch.sum(lse - chosen)


def cql_term(q: torch.Tensor, a: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    # E_s logsumexp(Q) - E_{(s,a)} Q(s,a)
    lse = masked_logsumexp(q, legal, dim=-1).mean()
    q_sa = q.gather(1, a.view(-1, 1)).squeeze(1).mean()
    return lse - q_sa


def soft_bellman_residual(q_s: torch.Tensor, q_ns: torch.Tensor, r: torch.Tensor, done: torch.Tensor, legal_ns: torch.Tensor, gamma: float, tau: float) -> torch.Tensor:
    # delta = Q(s,a) - (r + gamma * V(ns)) with V = tau * logsumexp(Q/tau)
    v_ns = tau * masked_logsumexp(q_ns / max(1e-6, tau), legal_ns, dim=-1)
    target = r + gamma * (1.0 - done) * v_ns
    return q_s - target


def estimate_beta_q(cfg: EstimatorConfig) -> str:
    logger = setup_logging(name="mdpmm.estimation")
    set_global_seeds(cfg.seed)
    device = torch.device(cfg.device)

    batch, s_dim, a_dim, pid_index = load_dataset(cfg.dataset_dir, device=str(device))

    # Model and optimizer
    model = MLPQ(s_dim, a_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Initialize betas to exp(mu)
    prior = BetaPrior(mu=cfg.beta_mu, sigma=cfg.beta_sigma)
    betas = {pid: float(math.exp(prior.mu)) for pid in pid_index.keys()}

    N = batch.s.shape[0]
    indices = torch.arange(N, device=device)

    def _rows_for_pids(pids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(batch.pid, dtype=torch.bool)
        for p in pids.tolist():
            mask = mask | (batch.pid == int(p))
        return torch.nonzero(mask, as_tuple=False).squeeze(1)

    logger.info(
        "Loaded dataset: N=%d, state_dim=%d, actions=%d, participants=%d",
        int(N), int(s_dim), int(a_dim), len(pid_index),
    )

    for rd in range(cfg.rounds):
        # E-step: update betas for all participants using current Q
        with torch.no_grad():
            q_all = model(batch.s)  # [N, A]
            policy_logits_all = _make_policy_logits(
                q_all,
                batch.legal_s,
                head=cfg.policy_head,
                normalize=cfg.adv_normalize,
                eps=cfg.adv_eps,
                clamp=cfg.adv_clip,
            )
        betas = update_betas_for_all(
            policy_logits_all,
            batch.a,
            batch.legal_s,
            pid_index,
            prior=prior,
            z0=prior.mu,
        )
        if cfg.debug:
            finite = [b for b in betas.values() if math.isfinite(b)]
            if cfg.policy_head == "adv":
                # Estimate average per-sample variance of advantages to diagnose identifiability.
                beta_values = torch.tensor(
                    [betas[int(p)] for p in batch.pid.tolist()],
                    dtype=torch.float32,
                    device=policy_logits_all.device,
                )
                logits = policy_logits_all
                pi = masked_softmax(beta_values.view(-1, 1) * logits, batch.legal_s, dim=-1)
                e_adv = torch.sum(pi * logits, dim=-1)
                e_adv2 = torch.sum(pi * (logits**2), dim=-1)
                var_adv = torch.mean(torch.clamp(e_adv2 - e_adv**2, min=0.0)).item()
            else:
                var_adv = float("nan")
            logger.info(
                "Round %d E-step: beta mean=%.4f std=%.4f min=%.4f max=%.4f var_logits=%.4f",
                rd,
                float(np.mean(finite)) if finite else float("nan"),
                float(np.std(finite)) if finite else float("nan"),
                float(np.min(finite)) if finite else float("nan"),
                float(np.max(finite)) if finite else float("nan"),
                var_adv,
            )

        # M-step: optimize Q with fixed betas
        for step in range(cfg.steps_per_round):
            # Mini-batch sample uniformly over rows
            idx = torch.randint(0, N, (min(cfg.batch_size, N),), device=device)
            s = batch.s[idx]
            ns = batch.ns[idx]
            a = batch.a[idx]
            r = batch.r[idx]
            done = batch.done[idx]
            legal_s = batch.legal_s[idx]
            legal_ns = batch.legal_ns[idx]
            # Map rows to betas
            beta_row = torch.tensor([betas[int(p)] for p in batch.pid[idx].tolist()], dtype=torch.float32, device=device)

            q = model(s)
            q_next = model(ns).detach()  # stop-grad on target for stability

            policy_logits = _make_policy_logits(
                q,
                legal_s,
                head=cfg.policy_head,
                normalize=cfg.adv_normalize,
                eps=cfg.adv_eps,
                clamp=cfg.adv_clip,
            )

            nll = behavior_nll(policy_logits, a, legal_s, beta_row) / max(1, len(idx))
            # Bellman residual on chosen a only
            q_sa = q.gather(1, a.view(-1, 1)).squeeze(1)
            bell = F.mse_loss(soft_bellman_residual(q_sa, q_next, r, done, legal_ns, cfg.gamma, cfg.tau), torch.zeros_like(q_sa))
            cql = cql_term(q, a, legal_s)

            loss = nll + cfg.lambda_bell * bell + cfg.lambda_cql * cql

            if not torch.isfinite(loss):
                if cfg.debug:
                    def _maybe_float(t: torch.Tensor) -> float:
                        val = t.detach().cpu()
                        return float(val) if torch.isfinite(val) else float("nan")

                    logger.warning(
                        "Skipping update due to non-finite loss (loss=%.3e, nll=%.3e, bell=%.3e, cql=%.3e)",
                        _maybe_float(loss),
                        _maybe_float(nll),
                        _maybe_float(bell),
                        _maybe_float(cql),
                    )
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if cfg.debug and (step % 50 == 0):
                logger.info(
                    "Round %d step %d | loss=%.4f nll=%.4f bell=%.4f cql=%.4f",
                    rd,
                    step,
                    float(loss.item()),
                    float(nll.item()),
                    float(bell.item()),
                    float(cql.item()),
                )

    # Final E-step for reporting
    with torch.no_grad():
        q_all = model(batch.s)
        policy_logits_all = _make_policy_logits(
            q_all,
            batch.legal_s,
            head=cfg.policy_head,
            normalize=cfg.adv_normalize,
            eps=cfg.adv_eps,
            clamp=cfg.adv_clip,
        )
    betas = update_betas_for_all(policy_logits_all, batch.a, batch.legal_s, pid_index, prior=prior, z0=prior.mu)

    # Write outputs
    from datetime import datetime

    run_dir = os.path.join(cfg.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Save Q
    torch.save({"state_dict": model.state_dict(), "state_dim": s_dim, "num_actions": a_dim, "hidden": cfg.hidden}, os.path.join(run_dir, "q.pt"))

    # Save beta estimates
    with open(os.path.join(run_dir, "beta_estimates.json"), "w", encoding="utf-8") as f:
        json.dump({"betas": betas, "prior": prior.__dict__}, f, indent=2)

    # Save manifest
    manifest = {
        "dataset_dir": os.path.abspath(cfg.dataset_dir),
        "seed": cfg.seed,
        "device": cfg.device,
        "rounds": cfg.rounds,
        "steps_per_round": cfg.steps_per_round,
        "lambda_bell": cfg.lambda_bell,
        "lambda_cql": cfg.lambda_cql,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "state_dim": s_dim,
        "num_actions": a_dim,
        "policy_head": cfg.policy_head,
        "adv_normalize": cfg.adv_normalize,
        "adv_eps": cfg.adv_eps,
        "adv_clip": cfg.adv_clip,
        "debug": cfg.debug,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return run_dir


def main() -> None:  # pragma: no cover (thin CLI)
    import argparse

    p = argparse.ArgumentParser(description="Estimate per-participant beta and shared Q from trajectories.jsonl")
    p.add_argument("--dataset-dir", required=True, help="Directory containing trajectories.jsonl")
    p.add_argument("--out-dir", default="artifacts/estimates")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps-per-round", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lambda-bell", type=float, default=0.5)
    p.add_argument("--lambda-cql", type=float, default=0.1)
    p.add_argument("--beta-mu", type=float, default=0.0)
    p.add_argument("--beta-sigma", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--policy-head", choices=["q", "adv"], default="q", help="Policy head for behavior likelihood.")
    p.add_argument(
        "--adv-normalize",
        choices=["none", "mean", "meanstd"],
        default="meanstd",
        help="Advantage normalization when --policy-head adv.",
    )
    p.add_argument("--adv-eps", type=float, default=0.05, help="Minimum per-state std when normalizing advantages.")
    p.add_argument(
        "--adv-clip",
        type=float,
        default=20.0,
        help="Clip normalized advantages into [-adv-clip, adv-clip]; set negative to disable.",
    )

    args = p.parse_args()
    cfg = EstimatorConfig(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        gamma=args.gamma,
        tau=args.tau,
        lambda_bell=args.lambda_bell,
        lambda_cql=args.lambda_cql,
        beta_mu=args.beta_mu,
        beta_sigma=args.beta_sigma,
        hidden=args.hidden,
        policy_head=args.policy_head,
        adv_normalize=args.adv_normalize,
        adv_eps=args.adv_eps,
        adv_clip=None if args.adv_clip is not None and args.adv_clip < 0 else args.adv_clip,
    )
    out = estimate_beta_q(cfg)
    print(out)


if __name__ == "__main__":  # pragma: no cover
    main()
