from __future__ import annotations

import json
from pathlib import Path

from mdpmm.inference.eval_beta import (
    load_true_betas_from_dataset,
    load_estimated_betas,
    evaluate_beta_recovery,
    save_beta_recovery_results,
)


def _write_toy_dataset(tmp: Path) -> Path:
    d = tmp / "toy_eval_ds"
    d.mkdir(parents=True, exist_ok=True)
    traj = d / "trajectories.jsonl"
    rows = [
        {"pid": 0, "episode": 0, "t": 0, "beta": 0.5, "s": [], "a": 0, "r": 0.0, "ns": [], "done": False, "legal_s": [True], "legal_ns": [True]},
        {"pid": 0, "episode": 0, "t": 1, "beta": 0.5, "s": [], "a": 0, "r": 0.0, "ns": [], "done": True, "legal_s": [True], "legal_ns": [True]},
        {"pid": 1, "episode": 0, "t": 0, "beta": 2.0, "s": [], "a": 0, "r": 0.0, "ns": [], "done": False, "legal_s": [True], "legal_ns": [True]},
    ]
    with traj.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def _write_estimates(tmp: Path, betas: dict[int, float]) -> Path:
    d = tmp / "toy_eval_estimates"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "beta_estimates.json").open("w", encoding="utf-8") as f:
        json.dump({"betas": betas}, f)
    return d


def test_beta_recovery_perfect(tmp_path: Path) -> None:
    ds = _write_toy_dataset(tmp_path)
    est = _write_estimates(tmp_path, {0: 0.5, 1: 2.0})
    tb = load_true_betas_from_dataset(str(ds))
    eb = load_estimated_betas(str(est))
    rows, metrics = evaluate_beta_recovery(tb, eb)
    assert metrics["n_overlap"] == 2
    assert abs(metrics["pearson_r"] - 1.0) < 1e-9
    assert abs(metrics["spearman_r"] - 1.0) < 1e-9
    assert metrics["mae"] == 0.0 and metrics["mape"] == 0.0
    out = tmp_path / "out_eval"
    paths = save_beta_recovery_results(str(out), rows, metrics, make_plots=False)
    assert (out / "beta_recovery.csv").exists()
    assert (out / "metrics.json").exists()

