#!/usr/bin/env bash
set -euo pipefail

# Estimate per-participant beta and shared Q from an offline dataset.
#
# Usage (defaults shown; override by exporting env vars before running):
#   DATASET_DIR=artifacts/datasets/peg4x4/peg4x4_logtraj \
#   OUT_DIR=artifacts/estimates \
#   ROUNDS=3 STEPS=200 LAMBDA_BELL=0.5 LAMBDA_CQL=0.1 GAMMA=0.99 TAU=1.0 \
#   BETA_MU=0.0 BETA_SIGMA=0.5 DEVICE=cpu HIDDEN=128 \
#   bash scripts/estimate_beta_q.sh
#
# To ignore reward (no Bellman consistency): set LAMBDA_BELL=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

# Defaults
DATASET_DIR="${DATASET_DIR:-artifacts/datasets/peg4x4/peg4x4_logtraj}"
OUT_DIR="${OUT_DIR:-artifacts/estimates}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ROUNDS="${ROUNDS:-100}"
STEPS="${STEPS:-200}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-1.0}"
LAMBDA_BELL="${LAMBDA_BELL:-0.5}"
LAMBDA_CQL="${LAMBDA_CQL:-0.1}"
BETA_MU="${BETA_MU:-0.0}"
BETA_SIGMA="${BETA_SIGMA:-0.5}"
HIDDEN="${HIDDEN:-128}"

echo "[estimate] PYTHONPATH: ${PYTHONPATH}"
echo "[estimate] dataset : ${DATASET_DIR}"
echo "[estimate] out_dir : ${OUT_DIR}"
echo "[estimate] rounds  : ${ROUNDS}, steps/round: ${STEPS}"
echo "[estimate] lambdas : bell=${LAMBDA_BELL}, cql=${LAMBDA_CQL}"
echo "[estimate] beta prior: mu=${BETA_MU}, sigma=${BETA_SIGMA}"
echo "[estimate] device  : ${DEVICE}, hidden: ${HIDDEN}"

if [[ ! -f "${DATASET_DIR}/trajectories.jsonl" ]]; then
  echo "[error] trajectories.jsonl not found under ${DATASET_DIR}" 1>&2
  exit 1
fi

python -m mdpmm.inference.estimate_beta_q \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --rounds "${ROUNDS}" \
  --steps-per-round "${STEPS}" \
  --gamma "${GAMMA}" \
  --tau "${TAU}" \
  --lambda-bell "${LAMBDA_BELL}" \
  --lambda-cql "${LAMBDA_CQL}" \
  --beta-mu "${BETA_MU}" \
  --beta-sigma "${BETA_SIGMA}" \
  --hidden "${HIDDEN}"

echo "[estimate] finished. See ${OUT_DIR}/<timestamp>/ for outputs."

