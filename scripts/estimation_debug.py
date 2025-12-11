#!/usr/bin/env python
from __future__ import annotations

"""
Lightweight debug runner for Beta–Q estimation that works without installing the package.

Usage examples:
  python scripts/estimation_debug.py --dataset-dir artifacts/datasets/synth/demo 
  python scripts/estimation_debug.py --dataset-dir <dir> --trace-pids 3 --print-trace
"""

import argparse
import json
import math
import os
import sys
from typing import List, Dict, Any


# Ensure local src/ is importable without pip install -e .
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

from mdpmm.inference.estimate_beta_q import (  # noqa: E402
    EstimatorConfig,
    estimate_beta_q,
    load_dataset,
    BetaPrior,
    MLPQ,
    _make_policy_logits,
)
from mdpmm.inference.diagnostics import trace_newton_updates, log_posterior_beta, newton_step_values  # noqa: E402


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
    p.add_argument(
        "--save-hessian",
        type=str,
        default="auto",
        help=(
            "Save per-participant Hessian/variance estimates for beta_j. "
            "Use a path to control location; 'auto' saves next to estimator run_dir "
            "when --run-estimator is set, otherwise under dataset_dir/hessian.json. "
            "Set to empty string to disable."
        ),
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

    def _hessian_record(tr) -> Dict[str, Any]:
        """Compute useful Hessian/variance diagnostics from a Newton trace."""
        last = tr[-1]
        H_z = float(last.H)
        var_z = -1.0 / H_z if H_z < 0 and math.isfinite(H_z) else None
        var_beta = -(last.beta**2) / H_z if H_z < 0 and math.isfinite(H_z) else None
        return {
            "z": float(last.z),
            "beta": float(last.beta),
            "g": float(last.g),
            "H_z": H_z,
            "var_z": var_z,
            "var_beta": var_beta,
            "obj": float(last.obj),
        }

    if args.trace_pids > 0:
        target_pids = sorted(pid_index.keys())[: args.trace_pids]
        for pid in target_pids:
            idxs: List[int] = pid_index[pid]
            tr = trace_newton_updates(q_all, batch.a, batch.legal_s, idxs, prior, z0=prior.mu, max_iter=5)
            rec = _hessian_record(tr)
            if args.print_trace:
                print(f"PID {pid} trace (len={len(tr)}):")
                for t in tr:
                    print(
                        f"  step={t.step:2d} z={t.z: .3f} beta={t.beta: .3f} g={t.g: .3e} H={t.H: .3e} obj={t.obj: .3f}"
                    )
                if rec["var_beta"] is not None:
                    print(
                        f"  H_z={rec['H_z']: .3e} var_z={rec['var_z']: .3e} var_beta={rec['var_beta']: .3e}"
                    )
                else:
                    print("  H_z not negative/finite; variance skipped")
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
        run_dir = estimate_beta_q(cfg)
        print(f"Estimator run_dir: {run_dir}")
    else:
        run_dir = None

    def _resolve_hessian_path() -> str | None:
        if not args.save_hessian or len(str(args.save_hessian)) == 0:
            return None
        if args.save_hessian == "auto":
            if run_dir:
                return os.path.join(run_dir, "hessian.json")
            return None
        return args.save_hessian

    def _compute_final_hessian(
        logits_all: torch.Tensor,
        betas: Dict[int, float],
        prior_obj: BetaPrior,
    ) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for pid, idxs in pid_index.items():
            beta_val = betas.get(pid)
            if beta_val is None or beta_val <= 0.0:
                continue
            z = math.log(beta_val)
            qj = logits_all[idxs]
            aj = batch.a[idxs]
            mj = batch.legal_s[idxs]
            g, H, _, _ = newton_step_values(qj, aj, mj, prior_obj, z)
            if math.isfinite(H) and H < 0:
                var_z = -1.0 / H
                var_beta = -(beta_val**2) / H
            else:
                var_z = None
                var_beta = None
            out[pid] = {
                "beta": float(beta_val),
                "z": float(z),
                "g": float(g),
                "H_z": float(H),
                "var_z": var_z,
                "var_beta": var_beta,
            }
        return out

    out_path = _resolve_hessian_path()
    if out_path and run_dir:
        # Load final model and betas to compute Hessian at the last estimate.
        ckpt_path = os.path.join(run_dir, "q.pt")
        betas_path = os.path.join(run_dir, "beta_estimates.json")
        if os.path.isfile(ckpt_path) and os.path.isfile(betas_path):
            ckpt = torch.load(ckpt_path, map_location=args.device)
            model = MLPQ(ckpt["state_dim"], ckpt["num_actions"], hidden=ckpt.get("hidden", 128))
            model.load_state_dict(ckpt["state_dict"])
            model.to(args.device)
            model.eval()
            with torch.no_grad():
                q_all_est = model(batch.s.to(args.device))
                policy_head = cfg.policy_head
                adv_norm = cfg.adv_normalize or ("meanstd" if policy_head == "adv" else "none")
                logits_all = _make_policy_logits(
                    q_all_est,
                    batch.legal_s.to(args.device),
                    head=policy_head,
                    normalize=adv_norm,
                    eps=cfg.adv_eps,
                    clamp=cfg.adv_clip,
                )
            with open(betas_path, "r", encoding="utf-8") as f:
                beta_data = json.load(f)
            betas_final = {int(k): float(v) for k, v in beta_data["betas"].items()}
            prior_final = BetaPrior(**beta_data.get("prior", prior.__dict__))
            hessian_out = _compute_final_hessian(logits_all, betas_final, prior_final)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"hessian": hessian_out, "prior": prior_final.__dict__}, f, indent=2)
            print(f"Saved final Hessian diagnostics to {out_path}")
        else:
            print(f"Skip saving Hessian: missing {ckpt_path} or {betas_path}")
    elif out_path and not run_dir:
        print("Hessian saving requested but --run-estimator was not run; nothing saved.")


if __name__ == "__main__":
    main()
