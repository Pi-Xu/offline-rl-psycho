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

## Train (Hydra)

Use Hydra + OmegaConf for fully config-driven training. Parameters are composed from `configs/` and can be overridden from the CLI.

Note: Hydra's run directory is set to `${paths.artifacts_dir}/${run_id}` (see `configs/config.yaml`).
This avoids a top-level `outputs/` folder — all logs/configs end up directly under the run folder alongside training artifacts.

### Hydra + OmegaConf

Compose configs and override from the command line. This enables splitting parameters across files (env, algo, train, eval, paths) and controlling them via CLI.

Run with Hydra entrypoint:

```bash
python -m mdpmm.training.hydra_train \
  env=peg7x7 algo=dqn \
  train.train_episodes=200 eval.eval_episodes=5 \
  paths.artifacts_dir=artifacts/models/peg/dqn \
  run_id=test_run
```

Config files live under `configs/`:

- `configs/config.yaml` – top-level defaults and switches
- `configs/env/*.yaml` – environment selection
- `configs/algo/*.yaml` – algorithm hyperparameters (e.g., DQN)
- `configs/train/*.yaml` – training loop settings
- `configs/eval/*.yaml` – evaluation settings
- `configs/paths/*.yaml` – artifact/output paths

Hydra supports overrides for any key, e.g. `algo.lr=1e-3 train.device=cuda`.


### Use a CNN Q-network

The default Q-network is an MLP. A lightweight CNN is also available and works for both 7x7 and 4x4 boards (single-channel; no extra mask inputs required).

- Switch via Hydra group:

```bash
python -m mdpmm.training.hydra_train algo=dqn_cnn
```

- Or override fields explicitly:

```bash
python -m mdpmm.training.hydra_train algo.model_type=cnn algo.cnn_channels='[16,32]' algo.cnn_hidden=256
```

Internally, the CNN reshapes the flattened observation `[H*W]` to `[1,H,W]`, applies a small Conv stack with global average pooling, then a linear head to action values.



## Artifacts
Saved under `artifacts/models/peg/dqn/<run_id>/`:

- `config.json` — resolved training config
- `metrics.jsonl` — per-step/episode/eval records
- `last.pt`, `best.pt` — model checkpoints (with meta). `best.pt` is selected by highest average return (tie-breakers: higher success rate, then fewer avg steps).
- `summary.json` — best and last eval summary

## Plot Metrics
Render per-episode and eval plots from a run directory:

```bash
python -m mdpmm.inference.plots --run-dir artifacts/models/peg/dqn/<run_id>
```

Alternatively, use the helper shell script (wraps the command above):

```bash
scripts/render_plots.sh --run-dir artifacts/models/peg/dqn/<run_id> --out-dir plots
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
  - `training/` — `hydra_train.py` (Hydra entrypoint), `train.py` (train loop)
  - `inference/plots.py` — plotting utility
  - `inference/generate.py` — generate offline trajectories from a checkpoint using softmax policy
  - `utils/` — config, logging, IO, seeding
- `configs/` — Hydra-config tree (`config.yaml`, `env/`, `algo/`, `train/`, `eval/`, `paths/`)
- `artifacts/` — run outputs (created at runtime)
- `doc/` — design notes and methodology drafts
- `tests/` — env/DQN/metrics tests
- `pyproject.toml` — dependencies and dev tooling

## Testing & Linting

```bash
pytest -q
ruff check . && black --check . && mypy src
```

## Debugging Without Installing

You can debug without `pip install -e .` by using the helper script that prepends `src/` to `sys.path`:

```bash
python scripts/estimation_debug.py --dataset-dir artifacts/datasets/synth/<name>
```

Options:
- `--trace-pids K` prints Newton traces for the first K participants: objective, gradient g, Hessian H, z/log-beta.
- `--run-estimator` launches the full estimator with `debug` logs enabled, printing loss decomposition (NLL/Bellman/CQL).

The script modifies `sys.path` at runtime so you can iterate without reinstalling.

## Beta Recovery Evaluation

Given a dataset with ground-truth betas and an estimation run directory, evaluate recovery quality:

```bash
python scripts/eval_beta_recovery.py \
  --dataset-dir artifacts/datasets/synth/<name> \
  --estimates-dir artifacts/estimates/<run_id>
```

Outputs under `<estimates-dir>/eval/` (or `--out-dir`):
- `beta_recovery.csv` — per-pid table: `pid,beta_true,beta_hat,abs_err,rel_err`
- `metrics.json` — `pearson_r`, `spearman_r`, `mae`, `mape`, counts
- `scatter_true_vs_hat.png` and `error_hist.png` (unless `--no-plots`)

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
- Add β estimation (MLE/EM) and evaluation harness for recovery on synthetic datasets.

## Generate Offline Data (β-aware)

Use a trained checkpoint (e.g., `best.pt`) to roll out trajectories with a softmax policy
π(a|s,β) ∝ exp{β Q(s,a)} over legal actions. This simulates participants with different decisiveness β.

CLI:

```bash
python -m mdpmm.inference.generate \
  --run-dir artifacts/models/peg/dqn/<run_id> \
  --participants 50 --episodes-per-participant 2 \
  --beta-mode lognormal --beta-mu 0.0 --beta-sigma 0.5 \
  --out-dir artifacts/datasets/synth --out-name demo
```

Or pass `--checkpoint-path <path/to/best.pt>` directly. Output files:

- `manifest.json` — config, β distribution, and checkpoint meta
- `trajectories.jsonl` — one record per transition with fields:
  `pid, episode, t, beta, s, a, r, ns, done, legal_s, legal_ns`.

Determinism: set `--seed` (also supported via `.env` SEED for defaults).

## Dataset Statistics

Given a generated dataset directory (containing `trajectories.jsonl` and `manifest.json`),
compute a beta histogram, per-participant average steps per episode, and a scatter plot of
beta vs. average steps:

```bash
python -m mdpmm.inference.stats \
  --dataset-dir artifacts/datasets/synth/<out_name>
```

Outputs to `<out_name>/plots/`:
- `participant_stats.csv` — columns: `pid,beta,episodes,avg_steps`
- `beta_hist.png` — histogram of betas
- `scatter_beta_vs_avg_steps.png` — scatter of beta vs avg steps
- `summary.json` — simple numeric summary
- Implement β MLE and evaluation harness for recovery on synthetic datasets.
- Add FastAPI inference server when needed.

## Beta–Q Estimation (psycho‑RL)

Estimate individual inverse temperatures (β) and a shared Q(s,a) from offline trajectories, following `doc/25_08_23_rl4psycho.md`. This implementation alternates:

- E-step: update per-participant β via 1D Newton on z=log β given current Q.
- M-step: update Q by minimizing behavior NLL with optional soft Bellman consistency (uses logged rewards r) and a light CQL term.

CLI:

```bash
python -m mdpmm.inference.estimate_beta_q \
  --dataset-dir artifacts/datasets/<name>/<run> \
  --rounds 3 --steps-per-round 200 \
  --lambda-bell 0.5 --lambda-cql 0.1 --gamma 0.99 --tau 1.0 \
  --beta-mu 0.0 --beta-sigma 0.5
```

Outputs in `artifacts/estimates/<timestamp>/`:
- `q.pt` — PyTorch state dict and metadata for the learned Q network
- `beta_estimates.json` — mapping `{pid: beta}` and the beta prior used
- `manifest.json` — run metadata

Data format reminder (one JSON per step):

```json
{"pid": 0, "episode": 0, "t": 0, "beta": 1.23,
 "s": [..], "a": 12, "r": -1.0,
 "ns": [..], "done": false,
 "legal_s": [true/false per action],
 "legal_ns": [true/false per action]}
```

Notes:
- This version does not perform inverse RL; it uses the logged reward `r` for the Bellman term. Set `--lambda-bell 0` to disable consistency if desired.
- Identification relies on the prior over β and the regularizers; very large λ can over-constrain Q.
