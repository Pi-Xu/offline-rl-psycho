
Train DQN on Peg Solitaire (epsilon-greedy)

- Command:
  - `python -m mdpmm.inference.cli train-dqn --config configs/train_dqn.yaml`
- Artifacts:
  - `artifacts/models/peg/dqn/<run_id>/best.pt` — best checkpoint by success rate.
  - `artifacts/models/peg/dqn/<run_id>/last.pt` — last checkpoint.
  - `metrics.jsonl` and `summary.json` for metrics.
