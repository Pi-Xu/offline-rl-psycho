from __future__ import annotations

from mdpmm.envs import make_env


def test_peg4x4_env_basics() -> None:
    env = make_env("peg4x4")
    obs, info = env.reset(seed=123)
    assert obs.shape == (16,)
    mask = env.legal_action_mask()
    assert mask.shape == (env.num_actions,)
    # There should be at least one legal move from the initial configuration
    assert mask.any()

