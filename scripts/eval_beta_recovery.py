#!/usr/bin/env python
from __future__ import annotations

"""
Evaluate beta recovery quality by comparing estimated betas against ground truth
stored in the dataset (trajectories.jsonl).

Outputs a CSV table, metrics.json, and optional plots in <out-dir>.

Usage:
  python scripts/eval_beta_recovery.py \
    --dataset-dir artifacts/datasets/synth/<name> \
    --estimates-dir artifacts/estimates/<run_id>
"""

import argparse
import math
import os
import sys


# Make local src/ importable without installation
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mdpmm.inference.eval_beta import (  # noqa: E402
    load_true_betas_from_dataset,
    load_estimated_betas,
    evaluate_beta_recovery,
    save_beta_recovery_results,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Beta recovery evaluation")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--estimates-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.estimates_dir, "eval")
    true_betas = load_true_betas_from_dataset(args.dataset_dir)
    est_betas = load_estimated_betas(args.estimates_dir)
    rows, metrics = evaluate_beta_recovery(true_betas, est_betas)
    # Save CSV + metrics first; we'll handle plots here to add annotations
    paths = save_beta_recovery_results(out_dir, rows, metrics, make_plots=False)

    if not args.no_plots and len(rows) > 0:
        # Local plotting with annotations for Pearson r and Spearman rho
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore

        xt = [r.beta_true for r in rows]
        yh = [r.beta_hat for r in rows]
        lo = min(min(xt), min(yh))
        hi = max(max(xt), max(yh))
        # Scatter with 45-degree line and annotation
        sc_path = os.path.join(out_dir, "scatter_true_vs_hat.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(xt, yh, alpha=0.75)
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("beta (true)")
        ax.set_ylabel("beta (estimated)")
        title = "Beta Recovery"
        ax.set_title(title)
        r = metrics.get("pearson_r")
        rho = metrics.get("spearman_r")
        r2 = metrics.get("r2")

        def _fmt(val: object) -> str:
            if isinstance(val, (int, float)) and math.isfinite(val):
                return f"{val:.3f}"
            return "nan"

        txt = f"r = {_fmt(r)}  |  rho = {_fmt(rho)}  |  R² = {_fmt(r2)}"
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        fig.tight_layout()
        fig.savefig(sc_path, dpi=300)
        pdf_path = sc_path.replace(".png", ".pdf")
        fig.savefig(pdf_path)
        plt.close(fig)
        paths["scatter"] = sc_path
        paths["scatter_pdf"] = pdf_path

        # Absolute error histogram (no annotation needed here)
        eh_path = os.path.join(out_dir, "error_hist.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist([abs(r.beta_hat - r.beta_true) for r in rows], bins=min(20, max(5, int(len(rows) ** 0.5))))
        ax.set_xlabel("|beta_hat - beta_true|")
        ax.set_ylabel("count")
        ax.set_title("Absolute Error Histogram")
        fig.tight_layout()
        fig.savefig(eh_path, dpi=300)
        eh_pdf_path = eh_path.replace(".png", ".pdf")
        fig.savefig(eh_pdf_path)
        plt.close(fig)
        paths["error_hist"] = eh_path
        paths["error_hist_pdf"] = eh_pdf_path

    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
