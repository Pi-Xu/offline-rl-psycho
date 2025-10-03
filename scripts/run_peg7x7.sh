#!/usr/bin/env bash
set -euo pipefail

# Simple, reproducible Peg Solitaire 7x7 experiment
# - Uses Hydra config composition
# - Writes artifacts under artifacts/models/peg/dqn/<run_id>/

# Allow running without installing the package by pointing PYTHONPATH to src/
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

RUN_ID="peg7x7_smoke_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] Starting peg7x7 run: ${RUN_ID}"
echo "[INFO] PYTHONPATH set to include: ${REPO_ROOT}/src"

# If you already activated a venv with deps installed, this will just run.
# Otherwise, follow README to install deps. (Torch install may require manual step.)

python -m mdpmm.training.hydra_train \
  env=peg7x7 algo=dqn \
  train.train_episodes=2000 train.max_steps_per_episode=80 \
  eval.eval_episodes=20 train.eval_every=40 \
  train.device=cuda train.print_eval=true \
  train.render_gif=true \
  train.seed=42 \
  algo.epsilon.decay_steps=75000 \
  algo.replay_capacity=25000 \
  algo.epsilon.end=0.05 \
  paths.artifacts_dir=artifacts/models/peg/dqn \
  run_id="${RUN_ID}"

echo "[INFO] Done. Artifacts in artifacts/models/peg/dqn/${RUN_ID}"
