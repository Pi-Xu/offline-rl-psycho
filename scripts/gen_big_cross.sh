#!/usr/bin/env bash
set -euo pipefail

# Generate offline rollouts for the big_cross board using a trained checkpoint.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

python -m mdpmm.inference.generate \
  --env-id big_cross \
  --run-dir artifacts/models/peg/dqn/big_cross_smoke_20251116_163240 \
  --device cuda \
  --participants 100 \
  --episodes-per-participant 50 \
  --max-steps-per-episode 100 \
  --seed 42 \
  --beta-mode lognormal \
  --beta-mu -0.5 \
  --beta-sigma 1.0 \
  --out-dir artifacts/datasets/big_cross \
  --out-name big_cross_logtraj \
  --model-type mlp

python -m mdpmm.inference.stats \
  --dataset-dir artifacts/datasets/big_cross/big_cross_logtraj
