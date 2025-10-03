from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib


def _safe_import_matplotlib_pyplot():
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt  # type: ignore

    return plt


@dataclass
class ParticipantStats:
    pid: int
    beta: float
    episodes: int
    avg_steps: float


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def compute_stats(dataset_dir: str) -> Tuple[List[ParticipantStats], Dict]:
    """Compute per-participant stats from trajectories.jsonl.

    Returns:
        (stats_list, meta)
    """
    traj_path = os.path.join(dataset_dir, "trajectories.jsonl")
    mani_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Not found: {traj_path}")
    meta: Dict = {}
    if os.path.exists(mani_path):
        try:
            with open(mani_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    # Gather per (pid, episode) step counts and per pid beta
    ep_steps: Dict[Tuple[int, int], int] = defaultdict(int)
    pid_betas: Dict[int, float] = {}
    for rec in _iter_jsonl(traj_path):
        pid = int(rec.get("pid"))
        ep = int(rec.get("episode"))
        ep_steps[(pid, ep)] += 1
        if pid not in pid_betas:
            try:
                pid_betas[pid] = float(rec.get("beta", float("nan")))
            except Exception:
                pid_betas[pid] = float("nan")

    # Reduce to per-pid averages
    steps_by_pid: Dict[int, List[int]] = defaultdict(list)
    for (pid, ep), n in ep_steps.items():
        steps_by_pid[pid].append(int(n))

    stats: List[ParticipantStats] = []
    for pid, counts in steps_by_pid.items():
        avg = float(np.mean(counts)) if counts else 0.0
        beta = float(pid_betas.get(pid, float("nan")))
        stats.append(ParticipantStats(pid=pid, beta=beta, episodes=len(counts), avg_steps=avg))

    stats.sort(key=lambda x: x.pid)
    return stats, meta


def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def save_stats_and_plots(dataset_dir: str, out_dir: str | None = None) -> Dict[str, str]:
    """Compute stats and save:
    - CSV summary (pid, beta, episodes, avg_steps)
    - beta_hist.png
    - scatter_beta_vs_avg_steps.png
    Returns map of artifact names to paths.
    """
    stats, meta = compute_stats(dataset_dir)
    if out_dir is None:
        out_dir = os.path.join(dataset_dir, "plots")
    _ensure_dir(out_dir)

    # Save CSV
    csv_path = os.path.join(out_dir, "participant_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "beta", "episodes", "avg_steps"])
        for s in stats:
            w.writerow([s.pid, s.beta, s.episodes, f"{s.avg_steps:.6f}"])

    # Prepare arrays for plotting
    betas = np.array([s.beta for s in stats if np.isfinite(s.beta)], dtype=float)
    avgs = np.array([s.avg_steps for s in stats if np.isfinite(s.beta)], dtype=float)

    plt = _safe_import_matplotlib_pyplot()

    # Beta histogram
    hist_path = os.path.join(out_dir, "beta_hist.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    if betas.size > 0:
        ax.hist(betas, bins=min(20, max(5, int(np.sqrt(len(betas)))))),
    ax.set_title("Beta Histogram")
    ax.set_xlabel("beta")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    # Scatter beta vs avg steps
    scatter_path = os.path.join(out_dir, "scatter_beta_vs_avg_steps.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    if betas.size > 0 and avgs.size == betas.size:
        ax.scatter(betas, avgs, alpha=0.7)
    ax.set_title("Beta vs Avg Steps")
    ax.set_xlabel("beta")
    ax.set_ylabel("avg steps per episode")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    # Save a small JSON summary
    json_summary_path = os.path.join(out_dir, "summary.json")
    try:
        summary = {
            "n_participants": len(stats),
            "beta_mean": float(np.mean(betas)) if betas.size > 0 else None,
            "beta_std": float(np.std(betas)) if betas.size > 0 else None,
            "avg_steps_mean": float(np.mean(avgs)) if avgs.size > 0 else None,
            "avg_steps_std": float(np.std(avgs)) if avgs.size > 0 else None,
            "meta": meta,
        }
        with open(json_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {
        "csv": csv_path,
        "beta_hist": hist_path,
        "scatter": scatter_path,
        "summary": json_summary_path,
    }


def main() -> None:  # pragma: no cover - thin CLI
    import argparse

    p = argparse.ArgumentParser(description="Compute stats and plots from generated trajectories")
    p.add_argument("--dataset-dir", required=True, help="Directory containing trajectories.jsonl and manifest.json")
    p.add_argument("--out-dir", default=None, help="Directory to save plots; default is <dataset-dir>/plots")
    args = p.parse_args()

    paths = save_stats_and_plots(args.dataset_dir, args.out_dir)
    # print paths for convenience
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":  # pragma: no cover
    main()

