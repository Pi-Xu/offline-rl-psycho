from __future__ import annotations

from typing import Callable, Dict

from .peg.peg_env import PegSolEnv
from .peg.boards import BOARD_SPECS
import numpy as np


_REGISTRY: Dict[str, Callable[[], PegSolEnv]] = {
    "peg7x7": lambda: PegSolEnv(),
    # Smaller 4x4 grid variant for faster experimentation
    # Initial empty is sampled from edge midpoints: (0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)
    "peg4x4": lambda: PegSolEnv(
        valid_mask=np.ones((4, 4), dtype=np.int8),
        initial_empty_choices=[
            (0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)
        ],
    ),
}

for board_id, spec in BOARD_SPECS.items():
    _REGISTRY[board_id] = lambda spec=spec: PegSolEnv(
        valid_mask=spec.valid_mask,
        initial_board=spec.initial_board,
    )


def make_env(env_id: str):
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown env_id: {env_id}. Available: {list(_REGISTRY)}")
    return _REGISTRY[env_id]()
