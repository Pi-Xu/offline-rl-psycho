from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    """Static board definition with -1 invalid cells and 0/1 legal cells."""

    name: str
    layout: np.ndarray
    description: str = ""
    valid_mask: np.ndarray = field(init=False, repr=False)
    initial_board: np.ndarray = field(init=False, repr=False)
    valid_cell_count: int = field(init=False)
    initial_peg_count: int = field(init=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.layout, dtype=np.int8)
        if arr.ndim != 2:
            raise ValueError("Board layout must be a 2D array")
        if not np.isin(arr, (-1, 0, 1)).all():
            raise ValueError("Board layout must contain only -1, 0, or 1")
        object.__setattr__(self, "layout", arr)
        valid_mask = (arr != -1).astype(np.int8)
        object.__setattr__(self, "valid_mask", valid_mask)
        board = np.where(arr == -1, 0, arr).astype(np.int8)
        object.__setattr__(self, "initial_board", board)
        object.__setattr__(self, "valid_cell_count", int(valid_mask.sum()))
        object.__setattr__(self, "initial_peg_count", int(board.sum()))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.layout.shape)


def _make_spec(name: str, rows: list[list[int]], description: str = "") -> BoardSpec:
    return BoardSpec(name=name, layout=np.array(rows, dtype=np.int8), description=description)


BOARD_SPECS: Dict[str, BoardSpec] = {
    "tiny_cross": _make_spec(
        "tiny_cross",
        rows=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        description="5x6 tiny cross board with offset leg",
    ),
    "big_cross": _make_spec(
        "big_cross",
        rows=[
            [-1, -1, 0, 0, 0, -1, -1],
            [-1, -1, 0, 1, 0, -1, -1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [-1, -1, 0, 1, 0, -1, -1],
            [-1, -1, 0, 0, 0, -1, -1],
        ],
        description="7x7 cross with invalid 2x2 corners",
    ),
    "big_L": _make_spec(
        "big_L",
        rows=[
            [1, 1, 1, -1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        description="4x4 board shaped like a block L",
    ),
    "diamond": _make_spec(
        "diamond",
        rows=[
            [-1, -1, 0, 0, 0, -1, -1],
            [-1, -1, 0, 1, 0, -1, -1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [-1, -1, 0, 1, 0, -1, -1],
            [-1, -1, 0, 0, 0, -1, -1],
        ],
        description="7x7 diamond with hollow center",
    ),
}
