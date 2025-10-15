from __future__ import annotations

"""
Utilities for diagnosing Beta–Q estimation.

Focus areas:
- Compute participant-specific log-posterior over z=log beta
- Produce Newton iteration traces (z, beta, g, H, obj)
- Quick sanity checks on dataset tensors

These helpers are import-light and CPU-friendly to keep debugging easy.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import math
import torch

from .estimate_beta_q import BetaPrior, masked_logsumexp


@dataclass
class NewtonTrace:
    step: int
    z: float
    beta: float
    g: float
    H: float
    obj: float


def log_posterior_beta(
    q_all: torch.Tensor,  # [N, A]
    a: torch.Tensor,  # [N]
    mask: torch.Tensor,  # [N, A] bool
    indices: List[int],
    prior: BetaPrior,
    z: float,
) -> float:
    """Participant log-posterior at z = log beta.

    lpost(z) = sum_t [ log pi_beta(a_t|s_t) ] - 0.5 * ((z-mu)/sigma)^2
    where pi_beta ∝ exp(beta * Q(s,a)) over legal actions.
    """
    if len(indices) == 0:
        return -0.5 * ((z - prior.mu) ** 2) / (prior.sigma**2)

    beta = math.exp(z)
    qj = q_all[indices]
    aj = a[indices]
    mj = mask[indices]
    logits = beta * qj
    lse = masked_logsumexp(logits, mj, dim=-1)  # [T]
    chosen = logits.gather(1, aj.view(-1, 1)).squeeze(1)  # [T]
    lp = torch.sum(chosen - lse).item()
    prior_term = -0.5 * ((z - prior.mu) ** 2) / (prior.sigma**2)
    return float(lp + prior_term)


def newton_step_values(
    qj: torch.Tensor,  # [T, A]
    aj: torch.Tensor,  # [T]
    mj: torch.Tensor,  # [T, A]
    prior: BetaPrior,
    z: float,
) -> Tuple[float, float, float, float]:
    """Compute (g, H, step, z_new) for one Newton update at z.

    Uses the same formulas as the estimator; helpful for step-by-step tracing.
    """
    beta = math.exp(z)
    logits = beta * qj
    pi = torch.where(mj, torch.exp(logits - masked_logsumexp(logits, mj, dim=-1).unsqueeze(-1)), torch.zeros_like(logits))
    e_q = torch.sum(pi * qj, dim=-1)
    e_q2 = torch.sum(pi * (qj**2), dim=-1)
    var_q = torch.clamp(e_q2 - e_q**2, min=0.0)
    q_chosen = qj.gather(1, aj.view(-1, 1)).squeeze(1)
    g = beta * torch.sum(q_chosen - e_q).item() - (z - prior.mu) / (prior.sigma**2)
    H = - (beta**2) * torch.sum(var_q).item() - 1.0 / (prior.sigma**2)
    if not math.isfinite(g) or not math.isfinite(H) or H >= 0.0:
        H = -max(1.0, abs(g))
    step = g / H
    z_new = z - step
    return float(g), float(H), float(step), float(z_new)


def trace_newton_updates(
    q_all: torch.Tensor,
    a: torch.Tensor,
    mask: torch.Tensor,
    indices: List[int],
    prior: BetaPrior,
    z0: float,
    max_iter: int = 10,
) -> List[NewtonTrace]:
    """Run a few Newton steps and record a compact trace for inspection."""
    if len(indices) == 0:
        return [NewtonTrace(step=0, z=prior.mu, beta=math.exp(prior.mu), g=0.0, H=-1.0, obj=log_posterior_beta(q_all, a, mask, indices, prior, prior.mu))]

    qj = q_all[indices]
    aj = a[indices]
    mj = mask[indices]
    z = float(z0)
    trace: List[NewtonTrace] = []
    for t in range(max_iter + 1):
        obj = log_posterior_beta(q_all, a, mask, indices, prior, z)
        g, H, step, z_new = newton_step_values(qj, aj, mj, prior, z)
        trace.append(NewtonTrace(step=t, z=z, beta=math.exp(z), g=g, H=H, obj=obj))
        z = z_new
    return trace


def quick_dataset_sanity(q: torch.Tensor, a: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Return a few simple diagnostics to catch bad states early."""
    return {
        "q_nan_frac": float(torch.isnan(q).float().mean().item()),
        "mask_true_frac": float(mask.float().mean().item()),
        "a_oob_frac": float(((a < 0) | (a >= q.shape[1])).float().mean().item()),
    }

