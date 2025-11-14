#!/usr/bin/env bash
set -euo pipefail

# Simple, reproducible Peg Solitaire 4x4 experiment
# - Uses Hydra config composition
# - Writes artifacts under artifacts/models/peg/dqn/<run_id>/

# Allow running without installing the package by pointing PYTHONPATH to src/
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

python -m mdpmm.inference.generate \
  --env-id peg7x7 \
  --run-dir artifacts/models/peg/dqn/peg7x7_smoke_20251015_224747 \
  --device cuda \
  --participants 100 \
  --episodes-per-participant 50 \
  --max-steps-per-episode 100 \
  --seed 42 \
  --beta-mode lognormal \
  --beta-mu -0.5 \
  --beta-sigma 1.0 \
  --out-dir artifacts/datasets/peg7x7 \
  --out-name peg7x7_logtraj \
  --model-type mlp

python -m mdpmm.inference.stats \
 --dataset-dir=artifacts/datasets/peg7x7/peg7x7_logtraj