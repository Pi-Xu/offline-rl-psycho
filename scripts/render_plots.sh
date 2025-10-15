#!/usr/bin/env bash

# Helper script to render plots from an inference/training run directory.
# Wraps `python -m mdpmm.inference.plots` and ensures src/ is on PYTHONPATH.

set -euo pipefail

function usage() {
  cat <<'EOF'
Usage: scripts/render_plots.sh --run-dir PATH [--out-dir NAME]

Arguments:
  --run-dir PATH   Directory containing metrics.jsonl (typically under artifacts/)
  --out-dir NAME   Optional subdirectory under the run dir to write plots (default: plots)

Examples:
  scripts/render_plots.sh --run-dir artifacts/models/peg/dqn/latest
  scripts/render_plots.sh --run-dir artifacts/models/peg/dqn/2025-10-01 --out-dir figures
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
SRC_PATH="${REPO_ROOT}/src"

RUN_DIR=""
OUT_DIR="plots"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "Error: --run-dir is required." >&2
  usage
  exit 1
fi

export PYTHONPATH="${SRC_PATH}:${PYTHONPATH:-}"

python -m mdpmm.inference.plots --run-dir "${RUN_DIR}" --out-dir "${OUT_DIR}"
