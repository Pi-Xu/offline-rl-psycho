# MDPMM — Peg Solitaire RL Baseline

Minimal offline RL scaffold featuring a DQN baseline on Peg Solitaire (7x7 and 4x4 variants), deterministic seeding, YAML config, JSONL metrics, and plotting utilities. Designed for reproducible experiments and extendable toward ability/temperature (beta) estimation workflows.

## Features
- Peg Solitaire environment with legal-action masks: `peg7x7` and `peg4x4`.
- DQN agent with target network, Huber loss, replay buffer, and gradient clipping.
- Deterministic runs via global seeding; artifacts saved under `artifacts/`.
- CLI training entrypoint + JSONL metrics + best/last checkpoints.
- Plotting tool to render progress from `metrics.jsonl`.
- Tests for env, DQN update, and metrics parsing.

## Requirements
- Python >= 3.9
- PyTorch (CPU by default). If install fails, follow platform-specific instructions on pytorch.org, then reinstall this package.

## Installation
Using uv (recommended):

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Using pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional environment variables: copy `.env.example` to `.env` and adjust

- `SEED` (default 42)
- `ARTIFACTS_DIR` (default `artifacts`)

## Train
With defaults:

```bash
python -m mdpmm.inference.cli train-dqn
```

With config:

```bash
python -m mdpmm.inference.cli train-dqn --config configs/train_dqn.yaml
```

### Key Config Fields (`configs/train_dqn.yaml`)
- Env: `env_id` (`peg7x7` or `peg4x4`), `max_steps_per_episode`, `train_episodes`
- DQN: `gamma`, `lr`, `batch_size`, `replay_capacity`, `learning_starts`, `target_update_interval`
- Exploration: `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`
- Misc: `seed`, `device` (`cpu`), `artifacts_dir`, `run_id` (optional)

## Artifacts
Saved under `artifacts/models/peg/dqn/<run_id>/`:

- `config.json` — resolved training config
- `metrics.jsonl` — per-step/episode/eval records
- `last.pt`, `best.pt` — model checkpoints (with meta)
- `summary.json` — best and last eval summary

## Plot Metrics
Render per-episode and eval plots from a run directory:

```bash
python -m mdpmm.inference.plots --run-dir artifacts/models/peg/dqn/<run_id>
```

Outputs PNGs in `<run_id>/plots/`:

- `episode_return.png`, `episode_steps.png`, `episode_epsilon.png`
- `eval_success_rate.png`, `eval_avg_steps.png`
- `train_*.png` (e.g., loss if present)
- `overview.png`

## Reproducibility
- Global seeds set for Python/NumPy/Torch (`mdpmm.utils.seeding.set_global_seeds`).
- Config-driven runs with resolved config saved alongside metrics.
- Deterministic env transitions; action legality enforced by masks.

## Repository Layout
- `src/mdpmm/`
  - `envs/` — Peg Solitaire (`peg_env.py`), registry (`make_env`)
  - `models/` — `dqn.py` (MLP Q-network, replay buffer, agent)
  - `inference/cli.py` — training entrypoint; `inference/plots.py` — plotting utility
  - `utils/` — config, logging, IO, seeding
- `configs/` — `train_dqn.yaml` and subdirs for future configs
- `artifacts/` — run outputs (created at runtime)
- `doc/` — design notes and methodology drafts
- `tests/` — env/DQN/metrics tests
- `pyproject.toml` — dependencies and dev tooling

## Testing & Linting

```bash
pytest -q
ruff check . && black --check . && mypy src
```

## Environments
- `peg7x7` — standard English cross board (center empty by default)
- `peg4x4` — compact variant for fast iteration

Observations: flattened board with invalid cells fixed at 0. Actions: enumerated jumps; legality provided as a boolean mask.

Rewards: per-step penalty (−1), large solved bonus (+100), additional terminal penalty proportional to remaining pegs if unsolved.

## Notes
- CPU is the default device; set `device` to `cuda` in the config if available.
- `metrics.jsonl` has kinds: `episode`, `train`, `eval`. See `tests/test_metrics_parsing.py` for expected structure.
- For background and model motivation, see docs in `doc/`.

## Roadmap (optional)
- Add offline trajectory generation with softmax policy over Q for β estimation.
- Implement β MLE and evaluation harness for recovery on synthetic datasets.
- Add FastAPI inference server when needed.

