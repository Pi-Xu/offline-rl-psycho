import pytest

from mdpmm.envs import make_env


BOARD_CASES = [
    (
        "tiny_cross",
        (5, 6),
        30,
        6,
        {
            (0, 0): 0,
            (2, 4): 1,  # (row 3, col 5) peg present
            (2, 5): 0,
        },
    ),
    (
        "big_cross",
        (7, 7),
        33,
        9,
        {
            (0, 0): -1,
            (3, 3): 1,
            (1, 3): 1,
        },
    ),
    (
        "big_L",
        (4, 4),
        15,
        14,
        {
            (0, 3): -1,
            (2, 1): 0,
            (3, 0): 1,
        },
    ),
    (
        "diamond",
        (7, 7),
        33,
        12,
        {
            (0, 0): -1,
            (3, 3): 0,
            (2, 3): 1,
        },
    ),
]


@pytest.mark.parametrize("env_id,shape,valid_count,peg_count,checks", BOARD_CASES)
def test_board_specs(env_id, shape, valid_count, peg_count, checks):
    env = make_env(env_id)
    assert env.valid_mask.shape == shape
    assert int(env.valid_mask.sum()) == valid_count

    obs, _ = env.reset(seed=123)
    board = obs.reshape(shape)
    assert int(board.sum()) == peg_count

    for (r, c), expected in checks.items():
        if expected == -1:
            assert env.valid_mask[r, c] == 0
            assert board[r, c] == 0
        else:
            assert env.valid_mask[r, c] == 1
            assert board[r, c] == expected
