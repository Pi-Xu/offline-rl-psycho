#!/usr/bin/env bash
set -euo pipefail

# Generate offline rollouts for the tiny_cross board using a trained checkpoint.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

python -m mdpmm.inference.generate \
  --env-id tiny_cross \
  --run-dir artifacts/models/peg/dqn/tiny_cross_smoke_20251116_163124 \
  --device cuda \
  --participants 200 \
  --episodes-per-participant 50 \
  --max-steps-per-episode 100 \
  --seed 42 \
  --beta-mode lognormal \
  --beta-mu 0.0 \
  --beta-sigma 0.75 \
  --out-dir artifacts/datasets/tiny_cross \
  --out-name tiny_cross_logtraj \
  --model-type mlp

python -m mdpmm.inference.stats \
  --dataset-dir artifacts/datasets/tiny_cross/tiny_cross_logtraj
