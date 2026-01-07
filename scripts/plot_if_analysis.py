from __future__ import annotations

"""
Quick visualization helpers for IF analysis outputs produced by scripts/if_analysis.py.

Given an IF analysis directory (containing group_step_summary.csv, pid_step_summary.csv),
this script renders a few concise PNG/PDF figures:
1) group_step_abs.(png|pdf)    — mean |IF| vs step for Low/High beta groups.
2) group_step_signed.(png|pdf) — mean IF vs step (directional) for groups (skip with --no-signed).
3) critical_step_hist.(png|pdf)— histogram of per-participant argmax |IF| step.
4) top_bottom_mean_abs(.png|.pdf) — mean |IF| curves for lowest/highest 10 betas
   (includes signed curves unless --no-signed). Optional _wp suffix if within-person stats exist.
5) step_abs_scatter_beta(.png|.pdf) — scatter of step vs mean |IF| colored by beta percentile.

Outputs are written next to the input CSVs.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import matplotlib
import numpy as np

# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_FORMATS = ("png", "pdf")
X_LABEL_STEP = "Step index"
Y_LABEL_MEAN_ABS = "Mean |IF|"
Y_LABEL_MEAN_SIGNED = "Mean IF (signed)"
Y_LABEL_PARTICIPANTS = "Participants"
Y_LABEL_MEAN_ABS_PER_PARTICIPANT = "Mean |IF| per participant"


def save_fig(fig: matplotlib.figure.Figure, out_dir: Path, stem: str, dpi: int = 200) -> None:
    for fmt in OUTPUT_FORMATS:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=dpi)


def load_betas_from_metadata(analysis_dir: Path) -> Tuple[Dict[int, float], Optional[List[float]]]:
    meta_path = analysis_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.json not found; cannot locate betas.")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    estimate_dir = Path(meta["estimate_dir"])
    beta_path = estimate_dir / "beta_estimates.json"
    if not beta_path.exists():
        raise FileNotFoundError(f"beta_estimates.json not found under {estimate_dir}")
    with beta_path.open("r", encoding="utf-8") as f:
        beta_file = json.load(f)
    betas = {int(k): float(v) for k, v in beta_file["betas"].items()}
    edges = meta.get("quantile_edges")
    return betas, edges


def load_group_curves(path: Path) -> Dict[str, Tuple[List[int], List[float], List[float], List[int]]]:
    """
    Returns mapping group -> (steps, mean_abs_if, mean_if, participants)
    """
    data: Dict[str, Tuple[List[int], List[float], List[float], List[int]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row["group"]
            if g not in data:
                data[g] = ([], [], [], [])
            steps, abs_if, mean_if, parts = data[g]
            steps.append(int(row["step"]))
            abs_if.append(float(row["mean_abs_if"]))
            mean_if.append(float(row["mean_if"]))
            parts.append(int(row["participants"]))
    # sort by step
    for g in list(data.keys()):
        steps, abs_if, mean_if, parts = data[g]
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        data[g] = (
            [steps[i] for i in order],
            [abs_if[i] for i in order],
            [mean_if[i] for i in order],
            [parts[i] for i in order],
        )
    return data


def load_pid_step(path: Path) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Returns pid -> {step: metrics}

    Required columns:
      - mean_if, mean_abs_if
    Optional columns:
      - mean_if_wp, mean_abs_if_wp
    """
    data: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["pid"])
            step = int(row["step"])
            metrics = {
                "mean_if": float(row["mean_if"]),
                "mean_abs_if": float(row["mean_abs_if"]),
            }
            if "mean_if_wp" in row and row["mean_if_wp"] != "":
                metrics["mean_if_wp"] = float(row["mean_if_wp"])
            if "mean_abs_if_wp" in row and row["mean_abs_if_wp"] != "":
                metrics["mean_abs_if_wp"] = float(row["mean_abs_if_wp"])
            data[pid][step] = metrics
    return data


def plot_group_curves(
    group_data: Dict[str, Tuple[List[int], List[float], List[float], List[int]]],
    out_dir: Path,
    *,
    include_signed: bool,
) -> None:
    colors = matplotlib.colormaps.get_cmap("tab10")

    # |IF|
    fig = plt.figure(figsize=(7, 4))
    for idx, (g, (steps, abs_if, _, _)) in enumerate(group_data.items()):
        plt.plot(steps, abs_if, marker="o", label=f"{g} |IF|", color=colors(idx % 10))
    plt.xlabel(X_LABEL_STEP)
    plt.ylabel(Y_LABEL_MEAN_ABS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_fig(fig, out_dir, "group_step_abs")
    plt.close(fig)

    if not include_signed:
        return

    # Signed IF
    fig = plt.figure(figsize=(7, 4))
    for idx, (g, (steps, _, mean_if, _)) in enumerate(group_data.items()):
        plt.plot(steps, mean_if, marker="o", label=f"{g} IF", color=colors(idx % 10))
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    plt.xlabel(X_LABEL_STEP)
    plt.ylabel(Y_LABEL_MEAN_SIGNED)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_fig(fig, out_dir, "group_step_signed")
    plt.close(fig)


def plot_critical_step_hist(pid_step: Dict[int, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    critical_steps: List[int] = []
    for _, step_dict in pid_step.items():
        if not step_dict:
            continue
        best_step = max(step_dict.items(), key=lambda kv: kv[1]["mean_abs_if"])[0]
        critical_steps.append(best_step)
    if not critical_steps:
        return
    fig = plt.figure(figsize=(6, 4))
    plt.hist(critical_steps, bins=range(min(critical_steps), max(critical_steps) + 2), color="#59a14f", edgecolor="white")
    plt.xlabel("Critical step (argmax |IF| per participant)")
    plt.ylabel(Y_LABEL_PARTICIPANTS)
    plt.tight_layout()
    save_fig(fig, out_dir, "critical_step_hist")
    plt.close(fig)


def _mean_curve_for_pids(
    pid_step: Dict[int, Dict[int, Dict[str, float]]],
    pids: List[int],
    *,
    key: str,
) -> Tuple[List[int], List[float]]:
    step_to_vals: Dict[int, List[float]] = defaultdict(list)
    for pid in pids:
        for step, metrics in pid_step.get(pid, {}).items():
            if key in metrics and np.isfinite(metrics[key]):
                step_to_vals[step].append(metrics[key])
    if not step_to_vals:
        return [], []
    steps = sorted(step_to_vals.keys())
    means = [float(np.mean(step_to_vals[s])) for s in steps]
    return steps, means


def plot_top_bottom_curves(
    pid_step: Dict[int, Dict[int, Dict[str, float]]],
    betas: Dict[int, float],
    out_dir: Path,
    *,
    include_signed: bool,
) -> None:
    if not betas:
        return
    sorted_pids = sorted(betas.items(), key=lambda kv: kv[1])
    bottom = [pid for pid, _ in sorted_pids[: min(10, len(sorted_pids))]]
    top = [pid for pid, _ in sorted_pids[-min(10, len(sorted_pids)) :]]

    def _plot(use_wp: bool) -> None:
        abs_key = "mean_abs_if_wp" if use_wp else "mean_abs_if"
        signed_key = "mean_if_wp" if use_wp else "mean_if"
        suffix = "_wp" if use_wp else ""

        steps_low_abs, mean_abs_low = _mean_curve_for_pids(pid_step, bottom, key=abs_key)
        steps_high_abs, mean_abs_high = _mean_curve_for_pids(pid_step, top, key=abs_key)
        if include_signed:
            steps_low_s, mean_s_low = _mean_curve_for_pids(pid_step, bottom, key=signed_key)
            steps_high_s, mean_s_high = _mean_curve_for_pids(pid_step, top, key=signed_key)
            fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
            ax0, ax1 = axes
        else:
            steps_low_s, mean_s_low, steps_high_s, mean_s_high = [], [], [], []
            fig, ax0 = plt.subplots(1, 1, figsize=(7, 4))
            ax1 = None

        if steps_low_abs:
            ax0.plot(steps_low_abs, mean_abs_low, marker="o", color="#4e79a7", label="Lowest 10 β")
        if steps_high_abs:
            ax0.plot(steps_high_abs, mean_abs_high, marker="o", color="#e15759", label="Highest 10 β")
        ax0.set_ylabel(Y_LABEL_MEAN_ABS)
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        if include_signed and ax1 is not None:
            if steps_low_s:
                ax1.plot(steps_low_s, mean_s_low, marker="o", color="#4e79a7", label="Lowest 10 β")
            if steps_high_s:
                ax1.plot(steps_high_s, mean_s_high, marker="o", color="#e15759", label="Highest 10 β")
            ax1.axhline(0.0, color="black", linewidth=1, alpha=0.6)
            ax1.set_xlabel(X_LABEL_STEP)
            ax1.set_ylabel(Y_LABEL_MEAN_SIGNED)
            ax1.grid(True, alpha=0.3)
        else:
            ax0.set_xlabel(X_LABEL_STEP)

        fig.tight_layout()
        save_fig(fig, out_dir, f"top_bottom_mean_abs{suffix}")
        plt.close(fig)

    _plot(use_wp=False)
    any_wp = any(
        ("mean_abs_if_wp" in m or "mean_if_wp" in m) for pid in pid_step.values() for m in pid.values()
    )
    if any_wp:
        _plot(use_wp=True)


def plot_scatter_beta(pid_step: Dict[int, Dict[int, Dict[str, float]]], betas: Dict[int, float], out_dir: Path) -> None:
    if not betas:
        return
    beta_items = sorted(betas.items(), key=lambda kv: kv[1])
    n = len(beta_items)
    beta_rank: Dict[int, float] = {}
    for rank_idx, (pid, _) in enumerate(beta_items):
        beta_rank[pid] = rank_idx / max(1, n - 1)  # percentile in [0,1]

    def _plot(use_wp: bool) -> None:
        xs: List[int] = []
        ys: List[float] = []
        ranks: List[float] = []
        y_key = "mean_abs_if_wp" if use_wp else "mean_abs_if"
        suffix = "_wp" if use_wp else ""

        for pid, step_dict in pid_step.items():
            if pid not in beta_rank:
                continue
            r = beta_rank[pid]
            for step, metrics in step_dict.items():
                if y_key not in metrics:
                    continue
                y = float(metrics[y_key])
                if not np.isfinite(y):
                    continue
                xs.append(step)
                ys.append(y)
                ranks.append(r)
        if not xs:
            return
        cmap = matplotlib.colormaps.get_cmap("plasma")
        fig = plt.figure(figsize=(7, 4))
        sc = plt.scatter(xs, ys, c=ranks, cmap=cmap, alpha=0.85, s=32, edgecolors="k", linewidths=0.25)
        plt.xlabel(X_LABEL_STEP)
        plt.ylabel(Y_LABEL_MEAN_ABS_PER_PARTICIPANT + (" [within-person normalized]" if use_wp else ""))
        cbar = plt.colorbar(sc)
        cbar.set_label("β percentile")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, out_dir, f"step_abs_scatter_beta{suffix}")
        plt.close(fig)

    _plot(use_wp=False)
    any_wp = any(("mean_abs_if_wp" in m) for pid in pid_step.values() for m in pid.values())
    if any_wp:
        _plot(use_wp=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot IF analysis summaries.")
    ap.add_argument("--analysis-dir", required=True, help="Path containing group_step_summary.csv and pid_step_summary.csv")
    ap.add_argument("--no-signed", action="store_true", help="Skip signed IF plots (directional).")
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir).resolve()
    group_path = analysis_dir / "group_step_summary.csv"
    pid_path = analysis_dir / "pid_step_summary.csv"
    if not group_path.exists() or not pid_path.exists():
        raise FileNotFoundError("Expected group_step_summary.csv and pid_step_summary.csv in analysis_dir")

    group_data = load_group_curves(group_path)
    pid_step = load_pid_step(pid_path)
    betas, edges = load_betas_from_metadata(analysis_dir)

    include_signed = not args.no_signed
    plot_group_curves(group_data, analysis_dir, include_signed=include_signed)
    plot_critical_step_hist(pid_step, analysis_dir)
    plot_top_bottom_curves(pid_step, betas, analysis_dir, include_signed=include_signed)
    plot_scatter_beta(pid_step, betas, analysis_dir)

    print(f"Plots written to {analysis_dir}")


if __name__ == "__main__":
    main()
