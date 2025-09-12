from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from mdpmm.utils.io import ensure_dir


def _read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole run
                continue
    return records


def parse_metrics_jsonl(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """
    Parse the metrics.jsonl file produced by training.

    Returns a dict with keys:
    - episodes: {episode: List[int], returns: List[float], steps: List[int], epsilon: List[float]}
    - evals: {episode: List[int], success_rate: List[float], avg_steps: List[float], avg_return: List[float]}
    - train: {step: List[int], <other metric names>: List[float]}  # optional
    """
    recs = _read_jsonl(path)
    episodes: Dict[str, List[Any]] = {"episode": [], "return": [], "steps": [], "epsilon": []}
    evals: Dict[str, List[Any]] = {
        "episode": [],
        "success_rate": [],
        "avg_steps": [],
        "avg_return": [],
    }
    train: Dict[str, List[Any]] = {"step": []}

    # collect all possible train metric keys aside from 'step' and 'kind'
    train_keys: List[str] = []

    for r in recs:
        kind = r.get("kind")
        if kind == "episode":
            episodes["episode"].append(int(r.get("episode", 0)))
            episodes["return"].append(float(r.get("return", 0.0)))
            episodes["steps"].append(int(r.get("steps", 0)))
            episodes["epsilon"].append(float(r.get("epsilon", 0.0)))
        elif kind == "eval":
            evals["episode"].append(int(r.get("episode", 0)))
            evals["success_rate"].append(float(r.get("success_rate", 0.0)))
            evals["avg_steps"].append(float(r.get("avg_steps", 0.0)))
            evals["avg_return"].append(float(r.get("avg_return", 0.0)))
        elif kind == "train":
            step = int(r.get("step", len(train.get("step", []))))
            train["step"].append(step)
            for k, v in r.items():
                if k in {"kind", "step"}:
                    continue
                if k not in train:
                    train[k] = []
                    train_keys.append(k)
                try:
                    train[k].append(float(v))
                except Exception:
                    # skip non-numeric
                    pass

    # Ensure consistency: if some train metric was missing for some steps, pad with nan
    max_len = len(train["step"]) if train["step"] else 0
    for k in list(train.keys()):
        if k == "step":
            continue
        if len(train[k]) < max_len:
            train[k] = train[k] + [np.nan] * (max_len - len(train[k]))

    return {"episodes": episodes, "evals": evals, "train": train}


def _safe_import_matplotlib():
    import matplotlib

    # Use a non-interactive backend for headless environments
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def _save_line_plot(
    x: Iterable[float],
    y: Iterable[float],
    out_path: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    marker: str | None = None,
    color: str | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    plt = _safe_import_matplotlib()
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(x), list(y), marker=marker or "", color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_overview(metrics: Dict[str, Any], out_path: str) -> None:
    plt = _safe_import_matplotlib()
    ensure_dir(os.path.dirname(out_path))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ep = metrics["episodes"]
    ev = metrics["evals"]

    # Episode return
    ax = axes[0, 0]
    ax.plot(ep["episode"], ep["return"], color="#1f77b4")
    ax.set_title("Episode Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Steps per episode
    ax = axes[0, 1]
    ax.plot(ep["episode"], ep["steps"], color="#ff7f0e")
    ax.set_title("Steps per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Eval success rate
    ax = axes[1, 0]
    if ev["episode"]:
        ax.plot(ev["episode"], ev["success_rate"], marker="o", color="#2ca02c")
        ax.set_ylim(0.0, 1.0)
    ax.set_title("Eval Success Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Eval avg steps
    ax = axes[1, 1]
    if ev["episode"]:
        ax.plot(ev["episode"], ev["avg_steps"], marker="o", color="#d62728")
    ax.set_title("Eval Avg Steps")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Steps")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_plots(run_dir: str, out_dir: str | None = None) -> List[str]:
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.jsonl not found in {run_dir}")
    metrics = parse_metrics_jsonl(metrics_path)

    out_root = Path(run_dir) / (out_dir or "plots")
    ensure_dir(out_root)

    saved: List[str] = []
    ep = metrics["episodes"]
    ev = metrics["evals"]
    tr = metrics["train"]

    # Episode-level plots
    if ep["episode"]:
        p = out_root / "episode_return.png"
        _save_line_plot(ep["episode"], ep["return"], str(p), title="Episode Return", xlabel="Episode", ylabel="Return")
        saved.append(str(p))

        p = out_root / "episode_steps.png"
        _save_line_plot(ep["episode"], ep["steps"], str(p), title="Steps per Episode", xlabel="Episode", ylabel="Steps")
        saved.append(str(p))

        p = out_root / "episode_epsilon.png"
        _save_line_plot(ep["episode"], ep["epsilon"], str(p), title="Epsilon Schedule", xlabel="Episode", ylabel="Epsilon")
        saved.append(str(p))

    # Eval-level plots
    if ev["episode"]:
        p = out_root / "eval_success_rate.png"
        _save_line_plot(ev["episode"], ev["success_rate"], str(p), title="Eval Success Rate", xlabel="Episode", ylabel="Success Rate", marker="o", ylim=(0.0, 1.0))
        saved.append(str(p))

        p = out_root / "eval_avg_steps.png"
        _save_line_plot(ev["episode"], ev["avg_steps"], str(p), title="Eval Avg Steps", xlabel="Episode", ylabel="Avg Steps", marker="o")
        saved.append(str(p))

    # Optional: training curves (e.g., loss) if present
    step_list = tr.get("step", [])
    for k, v in tr.items():
        if k in {"step"}:
            continue
        if not v:
            continue
        p = out_root / f"train_{k}.png"
        _save_line_plot(step_list, v, str(p), title=f"Train {k}", xlabel="Step", ylabel=k)
        saved.append(str(p))

    # Overview grid
    p = out_root / "overview.png"
    _save_overview(metrics, str(p))
    saved.append(str(p))

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render plots from a training run directory")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run dir containing metrics.jsonl")
    parser.add_argument("--out-dir", type=str, default="plots", help="Subdirectory under run dir to write plots")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved = render_plots(args.run_dir, args.out_dir)
    # Print saved paths for quick inspection
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()

