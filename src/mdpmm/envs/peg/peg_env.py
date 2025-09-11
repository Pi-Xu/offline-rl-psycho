from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


Coord = Tuple[int, int]
Action = Tuple[Coord, Coord, Coord]  # (from, over, to)


def _default_board_mask() -> np.ndarray:
    # English 7x7 cross: invalid corners
    mask = np.zeros((7, 7), dtype=np.int8)
    mask[2:5, :] = 1
    mask[:, 2:5] = 1
    return mask


def _enumerate_actions(valid_mask: np.ndarray) -> List[Action]:
    actions: List[Action] = []
    H, W = valid_mask.shape
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for r in range(H):
        for c in range(W):
            if not valid_mask[r, c]:
                continue
            for dr, dc in dirs:
                r_over, c_over = r + dr, c + dc
                r_to, c_to = r + 2 * dr, c + 2 * dc
                if 0 <= r_over < H and 0 <= c_over < W and 0 <= r_to < H and 0 <= c_to < W:
                    if valid_mask[r_over, c_over] and valid_mask[r_to, c_to]:
                        actions.append(((r, c), (r_over, c_over), (r_to, c_to)))
    return actions


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


class PegSolEnv:
    """Peg Solitaire 7x7 environment with discrete enumerated actions and masks.

    - Observation: flattened 49-d vector (invalid cells are always 0).
    - Action space: precomputed list of legal jump templates; per-state mask marks legal ones.
    - Reward: -1 per move; +100 bonus upon solving (single peg left).
    - Termination: solved (1 peg) or no legal moves or reaching max steps in episode control.
    """

    def __init__(self):
        self.valid_mask = _default_board_mask()  # 7x7
        self.actions: List[Action] = _enumerate_actions(self.valid_mask)
        self.num_actions: int = len(self.actions)
        self.obs_shape = (49,)
        self._board: np.ndarray | None = None
        self._steps = 0

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        board = self.valid_mask.astype(np.int8)
        # center empty
        board[3, 3] = 0
        self._board = board
        self._steps = 0
        return self._obs(), {"action_mask": self.legal_action_mask()}

    def legal_action_mask(self) -> np.ndarray:
        assert self._board is not None
        board = self._board
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        for idx, (src, over, dst) in enumerate(self.actions):
            sr, sc = src
            or_, oc = over
            dr, dc = dst
            if board[sr, sc] == 1 and board[or_, oc] == 1 and board[dr, dc] == 0:
                mask[idx] = True
        return mask

    def step(self, action: int) -> StepResult:
        assert self._board is not None
        legal = self.legal_action_mask()
        if action < 0 or action >= self.num_actions or not legal[action]:
            # illegal action: end episode with small penalty
            return StepResult(self._obs(), reward=-5.0, done=True, info={"illegal": True})

        src, over, dst = self.actions[action]
        sr, sc = src
        or_, oc = over
        dr, dc = dst
        self._board[sr, sc] = 0
        self._board[or_, oc] = 0
        self._board[dr, dc] = 1
        self._steps += 1

        done, solved = self._is_terminal()
        reward = -1.0
        if done and solved:
            reward += 100.0
        info = {"solved": solved, "action_mask": self.legal_action_mask()}
        return StepResult(self._obs(), reward=reward, done=done, info=info)

    def _obs(self) -> np.ndarray:
        assert self._board is not None
        # invalid cells are always 0 in obs
        return self._board.reshape(-1).astype(np.float32)

    def _is_terminal(self) -> Tuple[bool, bool]:
        assert self._board is not None
        pegs = int(self._board.sum())
        if pegs <= 1:
            return True, pegs == 1
        if not self.legal_action_mask().any():
            return True, False
        return False, False

