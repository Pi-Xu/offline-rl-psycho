from __future__ import annotations

import json
from pathlib import Path

from mdpmm.inference.plots import parse_metrics_jsonl


def test_parse_metrics_jsonl_basic(tmp_path: Path) -> None:
    # Create a small synthetic metrics.jsonl
    lines = [
        {"episode": 1, "kind": "episode", "return": -2.0, "steps": 10, "epsilon": 0.9},
        {"episode": 2, "kind": "episode", "return": -1.0, "steps": 12, "epsilon": 0.8},
        {"episode": 2, "kind": "eval", "success_rate": 0.2, "avg_steps": 50, "avg_return": -1.2},
        {"step": 100, "kind": "train", "loss": 0.5},
        {"step": 200, "kind": "train", "loss": 0.4},
    ]
    metrics_path = tmp_path / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")

    data = parse_metrics_jsonl(str(metrics_path))

    ep = data["episodes"]
    assert ep["episode"] == [1, 2]
    assert ep["steps"] == [10, 12]
    assert ep["epsilon"] == [0.9, 0.8]

    ev = data["evals"]
    assert ev["episode"] == [2]
    assert ev["success_rate"] == [0.2]
    assert ev["avg_steps"] == [50.0]
    assert ev["avg_return"] == [-1.2]

    tr = data["train"]
    assert tr["step"] == [100, 200]
    assert tr["loss"] == [0.5, 0.4]

