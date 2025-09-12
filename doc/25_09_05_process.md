
Train DQN on Peg Solitaire (epsilon-greedy)

- Command:
  - `python -m mdpmm.inference.cli train-dqn --config configs/train_dqn.yaml`
- Artifacts:
  - `artifacts/models/peg/dqn/<run_id>/best.pt` — best checkpoint by success rate.
  - `artifacts/models/peg/dqn/<run_id>/last.pt` — last checkpoint.
  - `metrics.jsonl` and `summary.json` for metrics.

Visualization

- After a run completes, render plots from metrics:
  - `python -m mdpmm.inference.plots --run-dir artifacts/models/peg/dqn/<run_id> --out-dir plots`
- Generated figures (PNG) under `<run_dir>/plots/`:
  - `episode_return.png`, `episode_steps.png`, `episode_epsilon.png`
  - `eval_success_rate.png`, `eval_avg_steps.png`
  - `train_<metric>.png` (if train-time metrics exist)
  - `overview.png` (2x2 summary grid)
