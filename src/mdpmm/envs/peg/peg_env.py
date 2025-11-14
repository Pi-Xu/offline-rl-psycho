from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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

    - Observation: flattened HxW vector (invalid cells are always 0).
    - Action space: precomputed list of legal jump templates; per-state mask marks legal ones.
    - Reward: per-step penalty (default −1). If solved (single peg left),
      add a large bonus (default +100). If the episode terminates unsolved,
      add an extra penalty proportional to the number of remaining pegs
      above 1 (default −5 per peg). This makes “leaving more pegs” strictly
      worse than taking a few extra steps to remove them.
    - Termination: solved (1 peg) or no legal moves or reaching max steps in episode control.
    """

    def __init__(
        self,
        *,
        valid_mask: np.ndarray | None = None,
        initial_empty: Tuple[int, int] | None = None,
        initial_empty_choices: Optional[List[Tuple[int, int]]] = None,
        step_penalty: float = -1.0,
        solved_bonus: float = 100.0,
        unsolved_penalty_per_peg: float = 5.0,
    ):
        self.valid_mask = valid_mask.copy() if valid_mask is not None else _default_board_mask()
        self.actions: List[Action] = _enumerate_actions(self.valid_mask)
        self.num_actions: int = len(self.actions)
        H, W = self.valid_mask.shape
        self.obs_shape = (H * W,)
        self._board: np.ndarray | None = None
        self._steps = 0
        # Initial empty slot; default to center
        if initial_empty is None:
            self._initial_empty = (H // 2, W // 2)
        else:
            self._initial_empty = initial_empty
        # Optional: a set of allowed initial empty coordinates to sample from at reset
        self._initial_empty_choices: Optional[List[Tuple[int, int]]] = None
        if initial_empty_choices is not None:
            # Validate provided choices fall within bounds and valid cells
            validated: List[Tuple[int, int]] = []
            for r, c in initial_empty_choices:
                if 0 <= r < H and 0 <= c < W and self.valid_mask[r, c] == 1:
                    validated.append((r, c))
            if not validated:
                raise ValueError("initial_empty_choices has no valid positions on the board")
            self._initial_empty_choices = validated
        # Reward shaping parameters
        self.step_penalty = float(step_penalty)
        self.solved_bonus = float(solved_bonus)
        # Applied at terminal if unsolved: penalty_per_peg * max(0, pegs_remaining - 1)
        self.unsolved_penalty_per_peg = float(unsolved_penalty_per_peg)

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        board = self.valid_mask.astype(np.int8)
        # initial empty (optionally sampled from allowed choices)
        if self._initial_empty_choices is not None and len(self._initial_empty_choices) > 0:
            idx = np.random.randint(0, len(self._initial_empty_choices))
            r0, c0 = self._initial_empty_choices[idx]
        else:
            r0, c0 = self._initial_empty
        if 0 <= r0 < board.shape[0] and 0 <= c0 < board.shape[1]:
            board[r0, c0] = 0
        self._board = board
        self._steps = 0
        return self._obs(), {"action_mask": self.legal_action_mask()}

    def reset_to_board(self, board: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to ``board`` after validation.

        ``board`` must match ``valid_mask`` in shape, use binary values {0,1},
        and have zeros in every invalid cell. The method copies the provided
        array to avoid external mutation.
        """

        arr = np.asarray(board, dtype=np.int8)
        if arr.shape != self.valid_mask.shape:
            raise ValueError(
                f"board shape {arr.shape} does not match valid_mask {self.valid_mask.shape}"
            )
        if not np.isin(arr, (0, 1)).all():
            raise ValueError("board must be binary (0 or 1 per cell)")
        invalid_mask = self.valid_mask == 0
        if np.any(arr[invalid_mask] != 0):
            raise ValueError("board places pegs on invalid cells")
        self._board = arr.copy()
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
        reward = self.step_penalty
        if done:
            if solved:
                reward += self.solved_bonus
            else:
                # Penalize proportionally to remaining pegs beyond 1
                pegs_remaining = int(self._board.sum())
                extra_pegs = max(0, pegs_remaining - 1)
                reward -= self.unsolved_penalty_per_peg * extra_pegs
        info = {"solved": solved, "action_mask": self.legal_action_mask()}
        return StepResult(self._obs(), reward=reward, done=done, info=info)

    # Small helper kept internal for testing and clarity
    def _unsolved_terminal_penalty(self, pegs_remaining: int) -> float:
        extra = max(0, pegs_remaining - 1)
        return self.unsolved_penalty_per_peg * extra

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
