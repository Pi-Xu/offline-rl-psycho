from __future__ import annotations

from mdpmm.envs import make_env


def test_unsolved_penalty_scales_with_remaining_pegs() -> None:
    env = make_env("peg7x7")
    # Default per-peg penalty is 5.0, step penalty magnitude is 1.0
    # Verify monotonicity: more pegs → larger terminal penalty
    p5 = env._unsolved_terminal_penalty(5)  # type: ignore[attr-defined]
    p10 = env._unsolved_terminal_penalty(10)  # type: ignore[attr-defined]
    assert p10 > p5 > 0

    # Each extra remaining peg should cost more than one extra step at default settings
    assert env.unsolved_penalty_per_peg > abs(env.step_penalty)

