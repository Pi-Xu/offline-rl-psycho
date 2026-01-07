from __future__ import annotations

"""
Step-level Influence Function (IF) analysis using precomputed beta/Q estimates.

Implements the absolute-time-step alignment described in PLAN_IF_Analysis.md:
- For each row (pid, episode, t), compute per-step IF ≈ g / I_j with
  g = beta_j * (Q_sa - E_pi[Q]) and I_j = beta_j^2 * sum_t Var_pi(Q) + 1/sigma^2.
- Aggregate by participant and absolute step index; then median-split participants
  into Low/High beta groups for group-level curves.

Outputs (under artifacts/if_analysis/<run_name>/ by default):
- if_rows.csv: per-decision diagnostics (pid, episode, t, g, var, IF, |IF|).
- pid_step_summary.csv: per-participant, per-step mean IF and |IF| with active episode count.
- group_step_summary.csv: Low/High group curves aggregated over participants.
- metadata.json: provenance and configuration used for this analysis.

This script is read-only w.r.t. model/data; it only writes analysis artifacts.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from mdpmm.inference.estimate_beta_q import BetaPrior, MLPQ, masked_softmax, _make_policy_logits


def load_estimate_run(estimate_dir: Path, device: torch.device):
    """Load Q, manifest, betas, prior."""
    manifest_path = estimate_dir / "manifest.json"
    betas_path = estimate_dir / "beta_estimates.json"
    q_path = estimate_dir / "q.pt"

    if not (manifest_path.exists() and betas_path.exists() and q_path.exists()):
        raise FileNotFoundError("estimate_dir must contain manifest.json, beta_estimates.json, q.pt")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    with betas_path.open("r", encoding="utf-8") as f:
        beta_file = json.load(f)
    betas: Dict[int, float] = {int(k): float(v) for k, v in beta_file["betas"].items()}
    prior_dict = beta_file.get("prior", {"mu": 0.0, "sigma": 1.0})
    prior = BetaPrior(mu=float(prior_dict.get("mu", 0.0)), sigma=float(prior_dict.get("sigma", 1.0)))

    ckpt = torch.load(q_path, map_location=device)
    model = MLPQ(state_dim=int(ckpt["state_dim"]), num_actions=int(ckpt["num_actions"]), hidden=int(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return manifest, betas, prior, model


def load_trajectories(dataset_dir: Path) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load trajectories.jsonl preserving pid/episode/t for alignment.

    Returns:
        meta: dict with lists for pid, episode, t, action indices.
        s: [N, S] float32
        legal_s: [N, A] bool
        a: [N] long
    """
    traj_path = dataset_dir / "trajectories.jsonl"
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectories.jsonl not found at {traj_path}")

    pids: List[int] = []
    episodes: List[int] = []
    steps: List[int] = []
    actions: List[int] = []
    states: List[List[float]] = []
    legal: List[List[bool]] = []

    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pids.append(int(row["pid"]))
            episodes.append(int(row.get("episode", 0)))
            steps.append(int(row.get("t", len(steps))))
            actions.append(int(row["a"]))
            states.append(row["s"])
            legal.append(row["legal_s"])

    s = torch.tensor(states, dtype=torch.float32)
    a = torch.tensor(actions, dtype=torch.long)
    legal_s = torch.tensor(legal, dtype=torch.bool)

    meta = {"pid": pids, "episode": episodes, "t": steps}
    return meta, s, legal_s, a


def compute_if_per_row(
    model: MLPQ,
    s: torch.Tensor,
    legal_s: torch.Tensor,
    a: torch.Tensor,
    betas: Dict[int, float],
    prior: BetaPrior,
    policy_cfg: Dict,
    pid_list: List[int],
    device: torch.device,
):
    """
    Compute g, var, IF, |IF| per row.
    """
    with torch.no_grad():
        s = s.to(device)
        legal_s = legal_s.to(device)
        a = a.to(device)
        q_all = model(s)  # [N, A]
        adv_clip = policy_cfg.get("adv_clip", None)
        adv_clip = None if adv_clip is not None and adv_clip < 0 else adv_clip
        policy_logits = _make_policy_logits(
            q_all,
            legal_s,
            head=policy_cfg["policy_head"],
            normalize=policy_cfg.get("adv_normalize", "meanstd"),
            eps=float(policy_cfg.get("adv_eps", 1e-6)),
            clamp=adv_clip,
        )

        beta_row = torch.tensor([betas[int(pid)] for pid in pid_list], dtype=torch.float32, device=device)
        scaled = beta_row.view(-1, 1) * policy_logits
        pi = masked_softmax(scaled, legal_s, dim=-1)

        e_q = torch.sum(pi * policy_logits, dim=-1)
        e_q2 = torch.sum(pi * (policy_logits**2), dim=-1)
        var_q = torch.clamp(e_q2 - e_q**2, min=0.0)

        q_chosen = policy_logits.gather(1, a.view(-1, 1)).squeeze(1)
        g = beta_row * (q_chosen - e_q)

    # Fisher per participant
    pid_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, pid in enumerate(pid_list):
        pid_to_indices[int(pid)].append(idx)

    denom_per_pid: Dict[int, float] = {}
    for pid, idxs in pid_to_indices.items():
        sum_var = float(var_q[idxs].sum().item())
        beta = float(betas[pid])
        denom_per_pid[pid] = beta * beta * sum_var + 1.0 / (prior.sigma**2)

    denom = torch.tensor([denom_per_pid[int(pid)] for pid in pid_list], dtype=torch.float32, device=device)
    if_val = g / denom

    out = {
        "g": g.cpu().numpy(),
        "var": var_q.cpu().numpy(),
        "if": if_val.cpu().numpy(),
        "abs_if": torch.abs(if_val).cpu().numpy(),
        "q_chosen": q_chosen.cpu().numpy(),
        "e_q": e_q.cpu().numpy(),
        "denom": denom.cpu().numpy(),
    }
    return out


def _within_person_scale(
    if_values: np.ndarray,
    pid_list: List[int],
    *,
    method: str,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute within-person normalized IF.

    We normalize the signed IF by a per-person scale computed from |IF|:
      - none: scale = 1
      - meanabs: scale = mean(|IF|)
      - maxabs: scale = max(|IF|)
      - zscore: signed z-score of IF (mean/std) and corresponding abs z-score

    Returns:
        if_wp: normalized signed IF
        abs_if_wp: normalized |IF|
    """
    if_values = np.asarray(if_values, dtype=np.float64)
    abs_vals = np.abs(if_values)

    if_wp = np.empty_like(if_values)
    abs_wp = np.empty_like(abs_vals)

    pid_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, pid in enumerate(pid_list):
        pid_to_indices[int(pid)].append(i)

    method = method.lower()
    if method not in {"none", "meanabs", "maxabs", "zscore"}:
        raise ValueError(f"Unknown within-person norm method: {method}")

    for pid, idxs in pid_to_indices.items():
        vals = if_values[idxs]
        abs_v = abs_vals[idxs]

        if method == "none":
            if_wp[idxs] = vals
            abs_wp[idxs] = abs_v
            continue

        if method in {"meanabs", "maxabs"}:
            scale = float(np.mean(abs_v) if method == "meanabs" else np.max(abs_v))
            scale = max(scale, eps)
            if_wp[idxs] = vals / scale
            abs_wp[idxs] = abs_v / scale
            continue

        # zscore
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        sd = max(sd, eps)
        if_wp[idxs] = (vals - mu) / sd
        # For abs values we z-score independently (magnitude distribution)
        mu_a = float(np.mean(abs_v))
        sd_a = float(np.std(abs_v))
        sd_a = max(sd_a, eps)
        abs_wp[idxs] = (abs_v - mu_a) / sd_a

    return if_wp.astype(np.float32), abs_wp.astype(np.float32)


def aggregate_by_step(meta: Dict, per_row: Dict) -> List[Dict]:
    """
    Aggregate |IF| and IF by participant & absolute step index.
    """
    pid_episode_map: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (pid, ep, t) in enumerate(zip(meta["pid"], meta["episode"], meta["t"])):
        pid_episode_map[int(pid)][int(ep)].append((t, idx))

    pid_step_rows: List[Dict] = []
    for pid, eps in pid_episode_map.items():
        # sort steps within each episode
        for ep in eps.values():
            ep.sort(key=lambda x: x[0])
        max_len = max(len(ep) for ep in eps.values())
        for step in range(max_len):
            step_indices: List[int] = []
            active_eps = 0
            for ep in eps.values():
                if len(ep) > step:
                    active_eps += 1
                    step_indices.append(ep[step][1])
            if active_eps == 0:
                continue
            mean_if = float(np.mean(per_row["if"][step_indices]))
            mean_abs_if = float(np.mean(per_row["abs_if"][step_indices]))
            mean_if_wp = float(np.mean(per_row["if_wp"][step_indices])) if "if_wp" in per_row else mean_if
            mean_abs_if_wp = (
                float(np.mean(per_row["abs_if_wp"][step_indices])) if "abs_if_wp" in per_row else mean_abs_if
            )
            pid_step_rows.append(
                {
                    "pid": pid,
                    "step": step,
                    "mean_if": mean_if,
                    "mean_abs_if": mean_abs_if,
                    "mean_if_wp": mean_if_wp,
                    "mean_abs_if_wp": mean_abs_if_wp,
                    "active_episodes": active_eps,
                }
            )
    return pid_step_rows


def aggregate_group_curves(
    pid_step_rows: List[Dict],
    betas: Dict[int, float],
    num_bins: int,
) -> Tuple[List[Dict], List[float]]:
    beta_values = np.array(list(betas.values()))
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(beta_values, quantiles).tolist()

    # Ensure strictly increasing edges by adding tiny epsilon if necessary
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    # Assign participants to bins
    bin_to_pids: Dict[int, List[int]] = defaultdict(list)
    for pid, b in betas.items():
        # last bin inclusive of upper edge
        for k in range(num_bins):
            left, right = edges[k], edges[k + 1]
            if (b >= left) and (b < right or (k == num_bins - 1 and b <= right)):
                bin_to_pids[k].append(pid)
                break

    rows = []
    for k in range(num_bins):
        pid_set = set(bin_to_pids.get(k, []))
        label = f"bin{k+1} [{edges[k]:.3g},{edges[k+1]:.3g}]"
        step_to_vals: Dict[int, List[float]] = defaultdict(list)
        step_to_dir: Dict[int, List[float]] = defaultdict(list)
        step_to_vals_wp: Dict[int, List[float]] = defaultdict(list)
        step_to_dir_wp: Dict[int, List[float]] = defaultdict(list)
        for row in pid_step_rows:
            if row["pid"] not in pid_set:
                continue
            step_to_vals[row["step"]].append(row["mean_abs_if"])
            step_to_dir[row["step"]].append(row["mean_if"])
            step_to_vals_wp[row["step"]].append(row["mean_abs_if_wp"])
            step_to_dir_wp[row["step"]].append(row["mean_if_wp"])
        for step, vals in step_to_vals.items():
            rows.append(
                {
                    "group": label,
                    "step": step,
                    "mean_abs_if": float(np.mean(vals)),
                    "mean_if": float(np.mean(step_to_dir[step])),
                    "mean_abs_if_wp": float(np.mean(step_to_vals_wp[step])) if step_to_vals_wp[step] else float("nan"),
                    "mean_if_wp": float(np.mean(step_to_dir_wp[step])) if step_to_dir_wp[step] else float("nan"),
                    "participants": len(vals),
                }
            )
    return rows, edges


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="IF analysis aligned by absolute time step.")
    p.add_argument("--estimate-dir", required=True, help="Path to an estimate run dir containing q.pt etc.")
    p.add_argument("--dataset-dir", help="Override dataset directory; default uses manifest dataset_dir.")
    p.add_argument("--out-dir", help="Output dir; default artifacts/if_analysis/<run_name>.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quantile-bins", type=int, default=2, help="Number of quantile bins for grouping (default: 2 = median split).")
    p.add_argument(
        "--within-person-norm",
        choices=["none", "meanabs", "maxabs", "zscore"],
        default="none",
        help="Optional within-person normalization for IF (helps cross-participant comparison).",
    )
    p.add_argument("--within-person-eps", type=float, default=1e-12, help="Epsilon for within-person normalization.")
    args = p.parse_args()

    estimate_dir = Path(args.estimate_dir).resolve()
    run_name = estimate_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts/if_analysis") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    manifest, betas, prior, model = load_estimate_run(estimate_dir, device=device)
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else Path(manifest["dataset_dir"]).resolve()

    meta, s, legal_s, a = load_trajectories(dataset_dir)
    per_row = compute_if_per_row(
        model=model,
        s=s,
        legal_s=legal_s,
        a=a,
        betas=betas,
        prior=prior,
        policy_cfg=manifest,
        pid_list=meta["pid"],
        device=device,
    )
    if_wp, abs_wp = _within_person_scale(
        per_row["if"],
        meta["pid"],
        method=args.within_person_norm,
        eps=float(args.within_person_eps),
    )
    per_row["if_wp"] = if_wp
    per_row["abs_if_wp"] = abs_wp

    # Row-level CSV
    row_csv = out_dir / "if_rows.csv"
    row_rows = []
    for i in range(len(meta["pid"])):
        row_rows.append(
            {
                "pid": meta["pid"][i],
                "episode": meta["episode"][i],
                "step": meta["t"][i],
                "action": int(a[i].item()),
                "g": float(per_row["g"][i]),
                "var": float(per_row["var"][i]),
                "if": float(per_row["if"][i]),
                "abs_if": float(per_row["abs_if"][i]),
                "if_wp": float(per_row["if_wp"][i]),
                "abs_if_wp": float(per_row["abs_if_wp"][i]),
                "beta": float(betas[int(meta["pid"][i])]),
                "denom": float(per_row["denom"][i]),
                "q_chosen": float(per_row["q_chosen"][i]),
                "e_q": float(per_row["e_q"][i]),
            }
        )
    write_csv(
        row_csv,
        row_rows,
        [
            "pid",
            "episode",
            "step",
            "action",
            "g",
            "var",
            "if",
            "abs_if",
            "if_wp",
            "abs_if_wp",
            "beta",
            "denom",
            "q_chosen",
            "e_q",
        ],
    )

    # Participant-step aggregation
    pid_step_rows = aggregate_by_step(meta, per_row)
    write_csv(
        out_dir / "pid_step_summary.csv",
        pid_step_rows,
        ["pid", "step", "mean_if", "mean_abs_if", "mean_if_wp", "mean_abs_if_wp", "active_episodes"],
    )

    # Group curves
    group_rows, edges = aggregate_group_curves(pid_step_rows, betas, num_bins=max(2, args.quantile_bins))
    write_csv(
        out_dir / "group_step_summary.csv",
        group_rows,
        ["group", "step", "mean_abs_if", "mean_if", "mean_abs_if_wp", "mean_if_wp", "participants"],
    )

    # Metadata
    meta_out = {
        "estimate_dir": str(estimate_dir),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "policy_head": manifest.get("policy_head"),
        "adv_normalize": manifest.get("adv_normalize"),
        "adv_eps": manifest.get("adv_eps"),
        "adv_clip": manifest.get("adv_clip"),
        "prior_mu": prior.mu,
        "prior_sigma": prior.sigma,
        "quantile_edges": edges,
        "within_person_norm": args.within_person_norm,
        "within_person_eps": float(args.within_person_eps),
        "notes": "Steps are zero-based t from trajectories.jsonl; IF uses Fisher approx g/(beta^2 Var_sum + 1/sigma^2).",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"IF analysis written to {out_dir}")


if __name__ == "__main__":
    main()
