import numpy as np

from mdpmm.envs import make_env
from mdpmm.models.dqn import DQNAgent, ReplayBuffer, Transition


def test_dqn_one_update():
    env = make_env("peg7x7")
    obs, info = env.reset(seed=0)
    agent = DQNAgent(obs_dim=49, num_actions=env.num_actions, lr=1e-3, gamma=0.99, device="cpu")
    buf = ReplayBuffer(100)
    # roll a few random steps to populate buffer
    for _ in range(10):
        legal = env.legal_action_mask()
        legal_idx = np.flatnonzero(legal)
        a = int(np.random.choice(legal_idx))
        sr = env.step(a)
        buf.add(
            Transition(
                s=obs,
                a=a,
                r=sr.reward,
                ns=sr.obs,
                done=sr.done,
                legal_mask_s=legal,
                legal_mask_ns=env.legal_action_mask(),
            )
        )
        obs = sr.obs
        if sr.done:
            obs, _ = env.reset(seed=0)
    batch = buf.sample(4)
    stats = agent.update(batch)
    assert "loss" in stats

