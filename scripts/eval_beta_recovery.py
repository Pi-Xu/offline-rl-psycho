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
import json
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
    p.add_argument(
        "--hessian-path",
        default=None,
        help="Optional path to hessian.json produced by estimation_debug/estimator. Defaults to <estimates-dir>/hessian.json if present.",
    )
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.estimates_dir, "eval")
    true_betas = load_true_betas_from_dataset(args.dataset_dir)
    est_betas = load_estimated_betas(args.estimates_dir)
    rows, metrics = evaluate_beta_recovery(true_betas, est_betas)
    # Save CSV + metrics first; we'll handle plots here to add annotations
    paths = save_beta_recovery_results(out_dir, rows, metrics, make_plots=False)

    print("metrics:")
    for key in sorted(metrics):
        val = metrics[key]
        if isinstance(val, (int, float)) and math.isfinite(val):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: nan")

    # Try to load Hessian diagnostics (optional)
    hessian_path = args.hessian_path or os.path.join(args.estimates_dir, "hessian.json")
    hessian = None
    if os.path.isfile(hessian_path):
        try:
            with open(hessian_path, "r", encoding="utf-8") as f:
                h_obj = json.load(f)
            raw = h_obj.get("hessian", {})
            hessian = {int(k): v for k, v in raw.items()}
        except Exception:
            hessian = None

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
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xt, yh, alpha=0.75)
        pad = max(1e-8, 0.05 * (hi - lo))
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
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

        # Log-log scatter with annotation on log-domain metrics
        log_sc_path = os.path.join(out_dir, "scatter_log_true_vs_hat.png")
        log_xt = [math.log(r.beta_true) for r in rows]
        log_yh = [math.log(r.beta_hat) for r in rows]
        log_lo = min(min(log_xt), min(log_yh))
        log_hi = max(max(log_xt), max(log_yh))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(log_xt, log_yh, alpha=0.75)
        log_pad = max(1e-8, 0.05 * (log_hi - log_lo))
        ax.plot([log_lo, log_hi], [log_lo, log_hi], color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(log_lo - log_pad, log_hi + log_pad)
        ax.set_ylim(log_lo - log_pad, log_hi + log_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("log beta (true)")
        ax.set_ylabel("log beta (estimated)")
        ax.set_title("Log Beta Recovery")
        log_r = metrics.get("log_pearson_r")
        log_rho = metrics.get("log_spearman_r")
        log_rmse = metrics.get("log_rmse")
        txt = f"log r = {_fmt(log_r)}  |  log rho = {_fmt(log_rho)}  |  log RMSE = {_fmt(log_rmse)}"
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
        fig.savefig(log_sc_path, dpi=300)
        log_pdf_path = log_sc_path.replace(".png", ".pdf")
        fig.savefig(log_pdf_path)
        plt.close(fig)
        paths["scatter_log"] = log_sc_path
        paths["scatter_log_pdf"] = log_pdf_path

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

        # Additional scatter plots when Hessian diagnostics are available
        def _hess_entry(pid: int):
            if hessian is None:
                return None
            return hessian.get(pid)

        def _widths_and_errors(log_scale: bool):
            xs = []
            ys = []
            for r in rows:
                h = _hess_entry(r.pid)
                if not h:
                    continue
                H = h.get("H_z")
                var_z = h.get("var_z")
                var_beta = h.get("var_beta")
                if H is None or not math.isfinite(H) or H >= 0:
                    if log_scale:
                        continue
                if log_scale:
                    var = var_z
                    if (var is None or not math.isfinite(var) or var <= 0) and H is not None and H < 0:
                        var = -1.0 / H
                    if var is None or not math.isfinite(var) or var <= 0:
                        continue
                    width = 3.92 * math.sqrt(var)
                    x = abs(math.log(r.beta_hat) - math.log(r.beta_true))
                else:
                    var = var_beta
                    if (var is None or not math.isfinite(var) or var <= 0) and H is not None and H < 0:
                        # delta method var_beta = -beta^2 / H
                        var = -(r.beta_hat**2) / H
                    if var is None or not math.isfinite(var) or var <= 0:
                        continue
                    width = 3.92 * math.sqrt(var)
                    x = abs(r.beta_hat - r.beta_true)
                xs.append(x)
                ys.append(width)
            return xs, ys

        if hessian is not None:
            # Beta scale
            xs, ys = _widths_and_errors(log_scale=False)
            if len(xs) > 0:
                ci_path = os.path.join(out_dir, "scatter_ci_beta.png")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(xs, ys, alpha=0.75)
                ax.set_xlabel("|beta_hat - beta_true|")
                ax.set_ylabel("CI width (beta) ≈ 3.92·sqrt(var_beta)")
                ax.set_title("Abs error vs CI width (beta)")
                fig.tight_layout()
                fig.savefig(ci_path, dpi=300)
                fig.savefig(ci_path.replace(".png", ".pdf"))
                plt.close(fig)
                paths["scatter_ci_beta"] = ci_path
                paths["scatter_ci_beta_pdf"] = ci_path.replace(".png", ".pdf")

            # Log-beta scale
            xs, ys = _widths_and_errors(log_scale=True)
            if len(xs) > 0:
                ci_log_path = os.path.join(out_dir, "scatter_ci_logbeta.png")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(xs, ys, alpha=0.75)
                ax.set_xlabel("|log beta_hat - log beta_true|")
                ax.set_ylabel("CI width (log beta) ≈ 3.92·sqrt(-1/H)")
                ax.set_title("Abs error vs CI width (log beta)")
                fig.tight_layout()
                fig.savefig(ci_log_path, dpi=300)
                fig.savefig(ci_log_path.replace(".png", ".pdf"))
                plt.close(fig)
                paths["scatter_ci_logbeta"] = ci_log_path
                paths["scatter_ci_logbeta_pdf"] = ci_log_path.replace(".png", ".pdf")

    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
