from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainDqnConfig(BaseModel):
    # Environment
    env_id: str = "peg7x7"
    max_steps_per_episode: int = 150
    train_episodes: int = 2000
    eval_episodes: int = 10
    eval_every: int = 50

    # DQN
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 50000
    learning_starts: int = 1000
    target_update_interval: int = 1000  # in steps

    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000

    # Misc
    seed: int = 42
    device: str = "cpu"
    artifacts_dir: str = "artifacts/models/peg/dqn"
    run_id: Optional[str] = None


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ARTIFACTS_DIR: str = Field(default="artifacts")
    SEED: int = Field(default=42)


def load_train_config(path: Optional[str]) -> TrainDqnConfig:
    if path is None:
        return TrainDqnConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TrainDqnConfig(**(data or {}))


def materialize_run_dir(base_dir: str, run_id: Optional[str]) -> str:
    from datetime import datetime

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(base_dir) / run_id
    os.makedirs(run_path, exist_ok=True)
    return str(run_path)

