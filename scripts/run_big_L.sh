#!/usr/bin/env bash
set -euo pipefail

# Simple, reproducible Peg Solitaire BIG-L experiment
# - Uses Hydra config composition
# - Writes artifacts under artifacts/models/peg/dqn/<run_id>/

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

RUN_ID="big_L_smoke_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] Starting big_L run: ${RUN_ID}"
echo "[INFO] PYTHONPATH set to include: ${REPO_ROOT}/src"

python -m mdpmm.training.hydra_train \
  env=big_L algo=dqn \
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
