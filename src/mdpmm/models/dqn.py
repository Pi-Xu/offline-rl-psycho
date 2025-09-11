from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPQ(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: bool
    legal_mask_s: np.ndarray
    legal_mask_ns: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        from collections import deque

        self.capacity = capacity
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:  # noqa: D401
        return len(self.buf)

    def add(self, tr: Transition) -> None:
        self.buf.append(tr)

    def sample(self, batch_size: int) -> Transition:
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        s = np.stack([self.buf[i].s for i in idx])
        a = np.array([self.buf[i].a for i in idx], dtype=np.int64)
        r = np.array([self.buf[i].r for i in idx], dtype=np.float32)
        ns = np.stack([self.buf[i].ns for i in idx])
        done = np.array([self.buf[i].done for i in idx], dtype=np.bool_)
        legal_s = np.stack([self.buf[i].legal_mask_s for i in idx])
        legal_ns = np.stack([self.buf[i].legal_mask_ns for i in idx])
        return Transition(s, a, r, ns, done, legal_s, legal_ns)


def masked_argmax(q: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    # q: [B, A], legal_mask: [B, A] bool
    neg_inf = torch.finfo(q.dtype).min
    masked_q = q.masked_fill(~legal_mask, neg_inf)
    return masked_q.argmax(dim=-1)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.q = MLPQ(obs_dim, num_actions).to(self.device)
        self.q_target = MLPQ(obs_dim, num_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.num_actions = num_actions

    @torch.no_grad()
    def act(self, obs: np.ndarray, legal_mask: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            legal_indices = np.flatnonzero(legal_mask)
            if len(legal_indices) == 0:
                return 0
            return int(np.random.choice(legal_indices))
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)  # [1, A]
        mask = torch.tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        a = masked_argmax(q, mask)
        return int(a.item())

    def update(self, batch: Transition) -> Dict[str, float]:
        s = torch.tensor(batch.s, dtype=torch.float32, device=self.device)
        a = torch.tensor(batch.a, dtype=torch.int64, device=self.device)
        r = torch.tensor(batch.r, dtype=torch.float32, device=self.device)
        ns = torch.tensor(batch.ns, dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        legal_ns = torch.tensor(batch.legal_mask_ns, dtype=torch.bool, device=self.device)

        q = self.q(s)  # [B, A]
        q_sa = q.gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q_target(ns)  # [B, A]
            a_next = masked_argmax(q_next, legal_ns)
            max_q_next = q_next.gather(1, a_next.view(-1, 1)).squeeze(1)
            target = r + (~done).float() * self.gamma * max_q_next

        loss = F.huber_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()
        return {"loss": float(loss.item())}

    def sync_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str, meta: Dict | None = None) -> None:
        obj = {
            "model_state": self.q.state_dict(),
            "meta": meta or {},
        }
        torch.save(obj, path)

