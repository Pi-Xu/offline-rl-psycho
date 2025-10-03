from __future__ import annotations

import json
from pathlib import Path

from mdpmm.inference.stats import compute_stats, save_stats_and_plots


def _write_line(p: Path, obj: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def test_stats_pipeline_minimal(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir(parents=True, exist_ok=True)
    traj = ds / "trajectories.jsonl"
    mani = ds / "manifest.json"

    # manifest
    mani.write_text(json.dumps({"env_id": "peg7x7", "seed": 0}), encoding="utf-8")

    # two participants, each one episode with 3 and 2 steps
    # pid 0, beta 0.5, episode 0: 3 steps
    for t in range(3):
        _write_line(traj, {"pid": 0, "episode": 0, "t": t, "beta": 0.5, "s": [], "a": 0, "r": 0, "ns": [], "done": t==2, "legal_s": [True], "legal_ns": [True]})
    # pid 1, beta 2.0, episode 0: 2 steps
    for t in range(2):
        _write_line(traj, {"pid": 1, "episode": 0, "t": t, "beta": 2.0, "s": [], "a": 0, "r": 0, "ns": [], "done": t==1, "legal_s": [True], "legal_ns": [True]})

    stats, meta = compute_stats(str(ds))
    assert len(stats) == 2
    d = {s.pid: s for s in stats}
    assert abs(d[0].avg_steps - 3.0) < 1e-6
    assert abs(d[1].avg_steps - 2.0) < 1e-6
    assert d[0].beta == 0.5 and d[1].beta == 2.0

    paths = save_stats_and_plots(str(ds))
    for v in paths.values():
        assert Path(v).exists()

