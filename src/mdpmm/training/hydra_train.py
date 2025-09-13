from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
import hydra

from mdpmm.training.train import train_dqn
from mdpmm.utils.config import build_train_config_from_hydra


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print resolved config for reproducibility
    print(OmegaConf.to_yaml(cfg, resolve=True))

    train_cfg = build_train_config_from_hydra(cfg)
    train_dqn(train_cfg)


if __name__ == "__main__":
    main()

