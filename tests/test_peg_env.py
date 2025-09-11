import numpy as np

from mdpmm.envs import make_env


def test_env_reset_and_mask():
    env = make_env("peg7x7")
    obs, info = env.reset(seed=123)
    assert obs.shape == (49,)
    mask = info["action_mask"]
    assert mask.shape == (env.num_actions,)
    assert mask.dtype == np.bool_
    assert mask.any(), "Initial state should have legal actions"


def test_step_changes_board():
    env = make_env("peg7x7")
    obs, info = env.reset(seed=123)
    mask = info["action_mask"]
    a = int(np.flatnonzero(mask)[0])
    sr = env.step(a)
    assert isinstance(sr.reward, float)
    assert sr.obs.shape == (49,)

