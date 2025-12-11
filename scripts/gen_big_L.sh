#!/usr/bin/env bash
set -euo pipefail

# Generate offline rollouts for the big_L board using a trained checkpoint.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

python -m mdpmm.inference.generate \
  --env-id big_L \
  --run-dir artifacts/models/peg/dqn/big_L_smoke_20251116_163334 \
  --device cuda \
  --participants 200 \
  --episodes-per-participant 50 \
  --max-steps-per-episode 100 \
  --seed 42 \
  --beta-mode lognormal \
  --beta-mu 0.0 \
  --beta-sigma 0.75 \
  --out-dir artifacts/datasets/big_L \
  --out-name big_L_logtraj \
  --model-type mlp

python -m mdpmm.inference.stats \
  --dataset-dir artifacts/datasets/big_L/big_L_logtraj
