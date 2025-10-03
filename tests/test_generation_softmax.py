from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mdpmm.inference.generate import _masked_softmax, generate_dataset
from mdpmm.models.dqn import DQNAgent
from mdpmm.envs import make_env
from mdpmm.utils.config import GenerateDataConfig


def test_masked_softmax_respects_mask():
    q = np.array([1.0, 2.0, 3.0, -1.0], dtype=np.float64)
    mask = np.array([True, False, True, False])
    p = _masked_softmax(q, mask, beta=2.0)
    # Probabilities only on legal indices 0 and 2
    assert p.shape == (4,)
    assert abs(p[1]) < 1e-12 and abs(p[3]) < 1e-12
    assert np.isclose(p[0] + p[2], 1.0)


def test_generate_dataset_with_temporary_checkpoint(tmp_path: Path):
    # Build a tiny, untrained agent and save a checkpoint
    env = make_env("peg7x7")
    obs_dim = int(np.prod(env.obs_shape))
    agent = DQNAgent(obs_dim=obs_dim, num_actions=env.num_actions, device="cpu")
    ckpt = tmp_path / "best.pt"
    agent.save(str(ckpt), meta={"test": True})

    cfg = GenerateDataConfig(
        env_id="peg7x7",
        checkpoint_path=str(ckpt),
        participants=2,
        episodes_per_participant=1,
        max_steps_per_episode=5,
        seed=123,
        beta_mode="fixed",
        beta_fixed=0.5,
        out_dir=str(tmp_path),
        out_name="gen",
    )
    out_dir = generate_dataset(cfg)
    traj = Path(out_dir) / "trajectories.jsonl"
    mani = Path(out_dir) / "manifest.json"
    assert traj.exists() and mani.exists()
    # Check at least one record line
    with traj.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        rec = json.loads(line)
    assert set(["pid", "episode", "t", "beta", "s", "a", "r", "ns", "done", "legal_s", "legal_ns"]).issubset(rec.keys())

