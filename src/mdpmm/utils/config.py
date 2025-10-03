from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any

try:
    from omegaconf import DictConfig
    from hydra.utils import to_absolute_path
except Exception:  # pragma: no cover - optional at runtime
    DictConfig = Any  # type: ignore
    def to_absolute_path(path: str) -> str:  # type: ignore
        return str(Path(path).absolute())


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
    # Optional: if provided (e.g., by Hydra), training will write into this directory directly
    run_dir: Optional[str] = None
    # Console printing options
    print_episode: bool = False
    episode_log_interval: int = 1
    print_eval: bool = True
    # Rendering options
    render_final_image: bool = True
    render_gif: bool = False
    gif_duration_ms: int = 400
    # Model
    model_type: str = "mlp"  # one of {mlp, cnn}
    cnn_channels: tuple[int, int] = (16, 32)
    cnn_hidden: int = 256


class GenerateDataConfig(BaseModel):
    # Model / env
    env_id: str = "peg7x7"
    # Either provide `checkpoint_path` directly, or point to a training `run_dir`
    checkpoint_path: Optional[str] = None
    run_dir: Optional[str] = None  # if set, uses `${run_dir}/best.pt` by default
    device: str = "cpu"
    # Rollout settings
    participants: int = 50  # number of synthetic participants (j)
    episodes_per_participant: int = 2  # number of episodes per participant
    max_steps_per_episode: int = 150
    seed: int = 42
    # Beta generation
    beta_mode: str = "lognormal"  # one of {lognormal, fixed}
    # log(beta) ~ N(mu, sigma^2)
    beta_mu: float = 0.0
    beta_sigma: float = 0.5
    # If fixed, use this value for all participants
    beta_fixed: float = 1.0
    # Output
    out_dir: str = "artifacts/datasets/synth"
    out_name: Optional[str] = None  # if None, auto from timestamp
    # Model architecture hints for restoring agent (should match training)
    model_type: str = "mlp"  # {mlp, cnn}
    cnn_channels: tuple[int, int] = (16, 32)
    cnn_hidden: int = 256

    def resolve_checkpoint(self) -> str:
        """Choose checkpoint file based on fields, prefer explicit path."""
        from pathlib import Path

        if self.checkpoint_path:
            return str(Path(self.checkpoint_path).expanduser())
        if self.run_dir:
            p = Path(self.run_dir) / "best.pt"
            return str(p)
        raise ValueError("Provide either `checkpoint_path` or `run_dir` to locate best.pt")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ARTIFACTS_DIR: str = Field(default="artifacts")
    SEED: int = Field(default=42)


def materialize_run_dir(base_dir: str, run_id: Optional[str]) -> str:
    from datetime import datetime

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(base_dir) / run_id
    os.makedirs(run_path, exist_ok=True)
    return str(run_path)


def build_train_config_from_hydra(cfg: DictConfig) -> TrainDqnConfig:
    """
    Convert a Hydra/OmegaConf config into our TrainDqnConfig.

    Expected structure:
      - cfg.env.env_id
      - cfg.algo.{gamma, lr, batch_size, replay_capacity, learning_starts, target_update_interval}
      - cfg.algo.epsilon.{start, end, decay_steps}
      - cfg.train.{max_steps_per_episode, train_episodes, eval_every, seed, device}
      - cfg.eval.eval_episodes
      - cfg.paths.artifacts_dir
      - cfg.run_id (optional)
      - cfg.hydra.run.dir (used as run_dir)
    """
    env_id = str(cfg.env.env_id)
    algo = cfg.algo
    eps = algo.get("epsilon", {})
    train = cfg.train
    eval_cfg = cfg.eval
    paths = cfg.paths

    artifacts_dir = to_absolute_path(str(paths.artifacts_dir))
    # Use Hydra's planned run directory to keep all outputs under artifacts/<run_id>
    try:
        hydra_run_dir = to_absolute_path(str(cfg.hydra.run.dir))  # type: ignore[attr-defined]
    except Exception:
        hydra_run_dir = None

    # Optional model options
    algo_model_type = str(getattr(algo, "model_type", "mlp")).lower()
    def _get_tuple2(x, default: tuple[int, int]) -> tuple[int, int]:
        try:
            if x is None:
                return default
            if isinstance(x, (list, tuple)) and len(x) >= 2:
                return (int(x[0]), int(x[1]))
        except Exception:
            pass
        return default

    cnn_channels = _get_tuple2(getattr(algo, "cnn_channels", None), (16, 32))
    cnn_hidden = int(getattr(algo, "cnn_hidden", 256))

    return TrainDqnConfig(
        env_id=env_id,
        max_steps_per_episode=int(train.max_steps_per_episode),
        train_episodes=int(train.train_episodes),
        eval_episodes=int(eval_cfg.eval_episodes),
        eval_every=int(train.eval_every),
        gamma=float(algo.gamma),
        lr=float(algo.lr),
        batch_size=int(algo.batch_size),
        replay_capacity=int(algo.replay_capacity),
        learning_starts=int(algo.learning_starts),
        target_update_interval=int(algo.target_update_interval),
        epsilon_start=float(eps.get("start", 1.0)),
        epsilon_end=float(eps.get("end", 0.05)),
        epsilon_decay_steps=int(eps.get("decay_steps", 50000)),
        seed=int(train.seed),
        device=str(train.device),
        artifacts_dir=artifacts_dir,
        run_id=(str(cfg.run_id) if cfg.get("run_id") is not None else None),
        run_dir=hydra_run_dir,
        print_episode=bool(getattr(train, "print_episode", False)),
        episode_log_interval=int(getattr(train, "episode_log_interval", 1)),
        print_eval=bool(getattr(train, "print_eval", True)),
        render_final_image=bool(getattr(train, "render_final_image", True)),
        render_gif=bool(getattr(train, "render_gif", False)),
        gif_duration_ms=int(getattr(train, "gif_duration_ms", 400)),
        model_type=algo_model_type,
        cnn_channels=cnn_channels,
        cnn_hidden=cnn_hidden,
    )
