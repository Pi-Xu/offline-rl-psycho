#!/usr/bin/env python
from __future__ import annotations

"""
Lightweight debug runner for Beta–Q estimation that works without installing the package.

Usage examples:
  python scripts/estimation_debug.py --dataset-dir artifacts/datasets/synth/demo 
  python scripts/estimation_debug.py --dataset-dir <dir> --trace-pids 3 --print-trace
"""

import argparse
import os
import sys
from typing import List


# Ensure local src/ is importable without pip install -e .
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

from mdpmm.inference.estimate_beta_q import (
    EstimatorConfig,
    estimate_beta_q,
    load_dataset,
    BetaPrior,
)
from mdpmm.inference.diagnostics import trace_newton_updates, log_posterior_beta  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Debug utilities for Beta–Q estimation")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--steps-per-round", type=int, default=50)
    # Estimator knobs
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lambda-bell", type=float, default=0.5)
    p.add_argument("--lambda-cql", type=float, default=0.1)
    p.add_argument("--beta-mu", type=float, default=0.0)
    p.add_argument("--beta-sigma", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--trace-pids", type=int, default=0, help="Trace Newton steps for first K participant IDs")
    p.add_argument("--print-trace", action="store_true", help="Print compact traces to stdout")
    p.add_argument("--run-estimator", action="store_true", help="Run full estimator after traces")
    p.add_argument(
        "--policy-head-adv",
        action="store_true",
        help="Enable advantage-based policy head for behavior likelihood.",
    )
    p.add_argument(
        "--adv-normalize",
        choices=["none", "mean", "meanstd"],
        default=None,
        help="Override advantage normalization strategy (default: meanstd when enabled).",
    )
    p.add_argument(
        "--adv-eps",
        type=float,
        default=0.05,
        help="Minimum per-state std when advantage normalization divides by std.",
    )
    p.add_argument(
        "--adv-clip",
        type=float,
        default=20.0,
        help="Clip normalized advantages into [-adv-clip, adv-clip]; set negative to disable.",
    )
    args = p.parse_args()

    batch, s_dim, a_dim, pid_index = load_dataset(args.dataset_dir, device=args.device)
    N = batch.s.shape[0]
    print(f"Loaded: N={N}, S={s_dim}, A={a_dim}, participants={len(pid_index)}")

    # Quick forward to get Q(s, a) from a tiny random MLP to exercise E-step traces without M-step noise
    with torch.no_grad():
        # Random small network just to produce q_all; real run will call estimator
        in_dim, out_dim = s_dim, a_dim
        q_all = torch.randn((N, out_dim), dtype=torch.float32)

    prior = BetaPrior()

    if args.trace_pids > 0:
        for pid in sorted(pid_index.keys())[: args.trace_pids]:
            idxs: List[int] = pid_index[pid]
            tr = trace_newton_updates(q_all, batch.a, batch.legal_s, idxs, prior, z0=prior.mu, max_iter=5)
            if args.print_trace:
                print(f"PID {pid} trace (len={len(tr)}):")
                for t in tr:
                    print(
                        f"  step={t.step:2d} z={t.z: .3f} beta={t.beta: .3f} g={t.g: .3e} H={t.H: .3e} obj={t.obj: .3f}"
                    )
            # Monotonic check (non-decreasing objective)
            objs = [x.obj for x in tr]
            mono = all(objs[i + 1] >= objs[i] - 1e-10 for i in range(len(objs) - 1))
            print(f"  monotonic_obj={mono}")

    if args.run_estimator:
        policy_head = "adv" if args.policy_head_adv else "q"
        adv_normalize = args.adv_normalize or ("meanstd" if policy_head == "adv" else "none")
        cfg = EstimatorConfig(
            dataset_dir=args.dataset_dir,
            device=args.device,
            rounds=max(1, args.rounds),
            steps_per_round=max(1, args.steps_per_round),
            debug=True,
            lr=args.lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            lambda_bell=args.lambda_bell,
            lambda_cql=args.lambda_cql,
            beta_mu=args.beta_mu,
            beta_sigma=args.beta_sigma,
            hidden=args.hidden,
            policy_head=policy_head,
            adv_normalize=adv_normalize,
            adv_eps=args.adv_eps,
            adv_clip=None if args.adv_clip is not None and args.adv_clip < 0 else args.adv_clip,
        )
        out_dir = estimate_beta_q(cfg)
        print(f"Estimator run_dir: {out_dir}")


if __name__ == "__main__":
    main()
