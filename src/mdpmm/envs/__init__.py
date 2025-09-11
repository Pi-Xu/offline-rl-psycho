from __future__ import annotations

from typing import Callable, Dict

from .peg.peg_env import PegSolEnv


_REGISTRY: Dict[str, Callable[[], PegSolEnv]] = {
    "peg7x7": lambda: PegSolEnv(),
}


def make_env(env_id: str):
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown env_id: {env_id}. Available: {list(_REGISTRY)}")
    return _REGISTRY[env_id]()

