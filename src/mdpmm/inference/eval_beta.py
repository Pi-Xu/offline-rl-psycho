from __future__ import annotations

"""
Beta recovery evaluation utilities.

Given a generated dataset directory containing `trajectories.jsonl` and an
estimation run directory containing `beta_estimates.json`, compute per-participant
errors and summary metrics, and optionally save plots.

This module is import-light and testable; a thin CLI wrapper lives in
scripts/eval_beta_recovery.py.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import csv
import json
import math
import os

import numpy as np


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_true_betas_from_dataset(dataset_dir: str) -> Dict[int, float]:
    """Extract per-participant true beta from trajectories.jsonl.

    Strategy: take the first encountered `beta` for each pid (constant per pid
    in our generators). If missing/NaN, pid is skipped.
    """
    traj = os.path.join(dataset_dir, "trajectories.jsonl")
    if not os.path.exists(traj):
        raise FileNotFoundError(f"Not found: {traj}")
    betas: Dict[int, float] = {}
    for rec in _iter_jsonl(traj):
        pid = int(rec.get("pid"))
        if pid in betas:
            continue
        try:
            b = float(rec.get("beta"))
            if math.isfinite(b) and b > 0:
                betas[pid] = b
        except Exception:
            continue
    return betas


def load_estimated_betas(estimates_dir: str) -> Dict[int, float]:
    """Load `{pid: beta_hat}` from <estimates_dir>/beta_estimates.json."""
    path = os.path.join(estimates_dir, "beta_estimates.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}. Pass the estimator's output directory containing beta_estimates.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    betas = obj.get("betas", obj)
    out: Dict[int, float] = {}
    for k, v in betas.items():
        try:
            pid = int(k)
            b = float(v)
            if math.isfinite(b) and b > 0:
                out[pid] = b
        except Exception:
            continue
    return out


def _ranks_with_ties(x: np.ndarray) -> np.ndarray:
    """Return average ranks (1..n) handling ties like SciPy."""
    n = x.size
    order = np.argsort(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        # find tie group
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average rank for i..j (1-based ranks)
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt((vx**2).sum() * (vy**2).sum())
    if denom <= 0:
        return float("nan")
    return float((vx * vy).sum() / denom)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _ranks_with_ties(x)
    ry = _ranks_with_ties(y)
    return _pearsonr(rx, ry)


@dataclass
class BetaEvalRow:
    pid: int
    beta_true: float
    beta_hat: float
    abs_err: float
    rel_err: float  # |hat-true| / true


def evaluate_beta_recovery(true_betas: Dict[int, float], est_betas: Dict[int, float]) -> Tuple[List[BetaEvalRow], dict]:
    """Join on pid, compute per-row errors and global metrics."""
    pids = sorted(set(true_betas.keys()) & set(est_betas.keys()))
    rows: List[BetaEvalRow] = []
    for pid in pids:
        bt = float(true_betas[pid])
        bh = float(est_betas[pid])
        ae = abs(bh - bt)
        re = (ae / bt) if bt > 0 else float("nan")
        rows.append(BetaEvalRow(pid, bt, bh, ae, re))

    x = np.array([r.beta_true for r in rows], dtype=float)
    y = np.array([r.beta_hat for r in rows], dtype=float)
    abs_err = np.array([r.abs_err for r in rows], dtype=float)
    rel_err = np.array([r.rel_err for r in rows if math.isfinite(r.rel_err)], dtype=float)

    if len(rows) >= 2:
        denom = float(((x - x.mean()) ** 2).sum())
        if denom > 0:
            r2 = float(1.0 - ((y - x) ** 2).sum() / denom)
        else:
            r2 = float("nan")
    else:
        r2 = float("nan")

    metrics = {
        "n_true": len(true_betas),
        "n_est": len(est_betas),
        "n_overlap": len(rows),
        "pearson_r": _pearsonr(x, y) if len(rows) >= 2 else float("nan"),
        "spearman_r": _spearmanr(x, y) if len(rows) >= 2 else float("nan"),
        "mae": float(abs_err.mean()) if len(rows) > 0 else float("nan"),
        "mape": float(rel_err.mean()) if rel_err.size > 0 else float("nan"),
        "median_abs_err": float(np.median(abs_err)) if len(rows) > 0 else float("nan"),
        "r2": r2,
    }
    return rows, metrics


def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _safe_import_matplotlib_pyplot():
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def save_beta_recovery_results(out_dir: str, rows: List[BetaEvalRow], metrics: dict, make_plots: bool = True) -> dict:
    out_dir = _ensure_dir(out_dir)
    # CSV
    csv_path = os.path.join(out_dir, "beta_recovery.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "beta_true", "beta_hat", "abs_err", "rel_err"])
        for r in rows:
            w.writerow([r.pid, f"{r.beta_true:.9f}", f"{r.beta_hat:.9f}", f"{r.abs_err:.9f}", f"{r.rel_err:.9f}"])

    # JSON metrics
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    paths = {"csv": csv_path, "metrics": json_path}

    if make_plots and len(rows) > 0:
        try:
            plt = _safe_import_matplotlib_pyplot()
            # Scatter: true vs hat
            sc_path = os.path.join(out_dir, "scatter_true_vs_hat.png")
            xt = [r.beta_true for r in rows]
            yh = [r.beta_hat for r in rows]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(xt, yh, alpha=0.75)
            lo = min(min(xt), min(yh))
            hi = max(max(xt), max(yh))
            ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("beta (true)")
            ax.set_ylabel("beta (estimated)")
            ax.set_title("Beta Recovery")
            fig.tight_layout()
            fig.savefig(sc_path, dpi=300)
            fig.savefig(sc_path.replace(".png", ".pdf"))
            plt.close(fig)
            paths["scatter"] = sc_path
            paths["scatter_pdf"] = sc_path.replace(".png", ".pdf")

            # Error histogram
            eh_path = os.path.join(out_dir, "error_hist.png")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist([r.abs_err for r in rows], bins=min(20, max(5, int(np.sqrt(len(rows))))) )
            ax.set_xlabel("|beta_hat - beta_true|")
            ax.set_ylabel("count")
            ax.set_title("Absolute Error Histogram")
            fig.tight_layout()
            fig.savefig(eh_path, dpi=300)
            fig.savefig(eh_path.replace(".png", ".pdf"))
            plt.close(fig)
            paths["error_hist"] = eh_path
            paths["error_hist_pdf"] = eh_path.replace(".png", ".pdf")
        except Exception:
            pass

    return paths
