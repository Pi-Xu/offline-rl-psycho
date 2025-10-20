from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mdpmm.inference.estimate_beta_q import (
    BetaPrior,
    TrajBatch,
    advantage_from_q,
    behavior_nll,
    estimate_beta_q,
    load_dataset,
    masked_softmax,
    update_betas_for_all,
    EstimatorConfig,
)


def _write_toy_dataset(tmp_path: Path) -> str:
    d = tmp_path / "toy"
    d.mkdir(parents=True, exist_ok=True)
    traj = d / "trajectories.jsonl"

    # Simple 1D state, 3 actions; legal are always all True for simplicity.
    rng = np.random.RandomState(0)
    A = 3
    def row(pid: int, t: int, s: float, a: int, r: float, ns: float, done: bool) -> dict:
        return {
            "pid": pid,
            "episode": 0,
            "t": t,
            "beta": float("nan"),
            "s": [float(s)],
            "a": int(a),
            "r": float(r),
            "ns": [float(ns)],
            "done": bool(done),
            "legal_s": [True] * A,
            "legal_ns": [True] * A,
        }

    with traj.open("w", encoding="utf-8") as f:
        # Participant 0 tends to choose a=0; participant 1 tends to choose a=2
        for t in range(20):
            s = rng.randn()
            f.write(json.dumps(row(0, t, s, 0, 0.0, s + 0.1, False)) + "\n")
        for t in range(20):
            s = rng.randn()
            f.write(json.dumps(row(1, t, s, 2, 0.0, s - 0.1, False)) + "\n")

    return str(d)


def test_masked_softmax_sanity() -> None:
    x = torch.tensor([[1.0, 0.0, -1.0]])
    mask = torch.tensor([[True, False, True]])
    p = masked_softmax(x, mask)
    assert torch.allclose(p.sum(dim=-1), torch.tensor([1.0]))
    assert p[0, 1] == 0.0


def test_advantage_meanstd_single_legal_grad_finite() -> None:
    q = torch.tensor([[0.1, 0.2, -0.3], [10.0, 10.0, 10.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False], [True, False, False]])
    out = advantage_from_q(q, legal, normalize="meanstd", eps=0.05, clamp=20.0)
    out.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_e_step_updates_beta_direction(tmp_path: Path) -> None:
    # Build a tiny dataset and a fixed Q table to drive betas
    ds = _write_toy_dataset(tmp_path)
    batch, s_dim, a_dim, pid_index = load_dataset(ds)
    assert a_dim == 3 and s_dim == 1

    # Fixed Q makes action 0 attractive for participant 0's states
    with torch.no_grad():
        # Encode preference via Q(s,a) independent of s for simplicity
        q = torch.zeros((batch.s.shape[0], a_dim), dtype=torch.float32)
        q[:, 0] = 1.0  # action 0 highest
        # participant 1 prefers action 2: swap their rows
        idx_p1 = torch.nonzero(batch.pid == 1, as_tuple=False).squeeze(1)
        q[idx_p1, 2] = 1.0
        q[idx_p1, 0] = 0.0

    prior = BetaPrior(mu=0.0, sigma=1.0)
    betas = update_betas_for_all(q, batch.a, batch.legal_s, pid_index, prior)
    # Expect finite, positive betas
    assert all(b > 0 and np.isfinite(b) for b in betas.values())
    # Both participants consistently pick the greedy action under Q, so beta should be > exp(mu)=1
    assert betas[0] > 1.0 and betas[1] > 1.0


def test_estimator_smoke_runs(tmp_path: Path) -> None:
    ds = _write_toy_dataset(tmp_path)
    out = tmp_path / "outs"
    cfg = EstimatorConfig(dataset_dir=ds, out_dir=str(out), rounds=1, steps_per_round=5, batch_size=16)
    run_dir = estimate_beta_q(cfg)
    assert Path(run_dir).exists()
    assert (Path(run_dir) / "q.pt").exists()
    assert (Path(run_dir) / "beta_estimates.json").exists()
