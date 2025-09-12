Objective

- Train a DQN-like agent in Peg Solitaire, generate offline trajectories with a softmax policy over Q, and estimate the inverse temperature parameter
beta from the collected data using maximum likelihood.

Scope

- Environment: standard 7x7 Peg Solitaire (center empty), deterministic transitions.
- Agent: DQN/DoubleDQN with target network and replay buffer.
- Policy: softmax over Q-values, πβ(a|s) ∝ exp(β·Q(s,a)) over legal actions.
- Data: store transitions and Q-snapshots; generate multiple datasets with known β_true.
- Estimation: MLE of β using collected (s, a, legal actions, Q(s,·)) tuples.

Non-Goals (initial)

- Complex function approximation or large models.
- Advanced offline RL (CQL/IQL) or OPE beyond basic checks.
- Complex priors/hierarchical β (can be added later).

System Overview

- Environment: PegEnv (Gym-compatible), exposes obs and action_mask.
- Agent: DQN with MLP, Huber loss, target sync, epsilon for exploration off by default (use softmax with β during rollouts).
- Data Generator: runs policy with fixed β_true to create reproducible datasets.
- Estimator: β MLE with gradient/Hessian; returns β_hat and standard error.
- Evaluation: online success rate, average steps/reward; β recovery on synthetic data.
- Config/Logging: YAML configs with pydantic settings, deterministic seeds, structured logs, artifacts persisted.

Environment (Peg Solitaire)

- Observation: 7x7 board flattened to length-49 vector; invalid slots masked.
- Action space: enumerated jumps (from, over, to) in four directions; mask illegal actions.
- Reward: per-step −1; solve +100. If the episode ends unsolved, add an
  extra penalty proportional to remaining pegs above 1 (default −5 per peg),
  so leaving more pegs is worse than taking a few extra steps. Discount γ fixed (e.g., 0.99).
- Termination: 1 peg left or no legal actions; optional max steps.
- Determinism: reset(seed=...); stable reward scale for Q identifiability.

Agent (DQN)

- Network: small MLP on 49-d input → |A| outputs.
- Training: replay buffer, target network, Huber loss, Adam optimizer, periodic eval.
- Behavior policy: softmax with β_train for stability; for data generation use fixed β_true.
- Logging: success rate, average steps/reward, loss, TD error; checkpoint best model.

Offline Data Generation

- Policy: use learned Q and softmax with β_true ∈ {1,2,5,10} to produce datasets.
- Record per step: obs, action, reward, next_obs, done, legal_actions, q_values (for legal actions), episode_id, t, and optional info.
- Storage: Parquet (preferred) or JSONL under artifacts/datasets/peg/<run_id>/.
- Splits: episode-level train/val/test (e.g., 80/10/10).
- Dataset card: JSON with config, seed, counts, success rate, avg length, hash.

Beta Estimation (MLE)

- Model: action probabilities follow softmax over Q restricted to legal actions.
- Likelihood (per sample i): l_i(β) = β·Q_i(a_i) − log Σ_{b∈legal_i} exp(β·Q_i(b)).
- Objective: maximize L(β) = Σ_i l_i(β), β ≥ 0.
- Gradient: dL/dβ = Σ_i [Q_i(a_i) − E_{πβ,i}[Q_i(·)]].
- Curvature: d2L/dβ2 = −Σ_i Var_{πβ,i}[Q_i(·)] ≤ 0 (strictly concave unless degenerate).
- Optimization: 1D Newton with line search or Brent on [0, β_max]; return β_hat and SE = sqrt(−1 / d2L/dβ2 at β_hat).
- Diagnostics: plot L(β) shape, compare β_hat vs β_true, report CI and goodness-of-fit.

Reproducibility

- Seeds: set SEED across random, numpy, torch; optionally torch.use_deterministic_algorithms(True).
- Config: configs/train_dqn.yaml, configs/gen_data.yaml, configs/estimate_beta.yaml.
- Artifacts: models, datasets, metrics written to artifacts/ with timestamps and config hashes.
- Dataset card and run metadata written alongside artifacts for traceability.

Metrics and Acceptance

- Agent: success rate, average steps, average reward; agent better than random.
- Data: reproducible counts and stats under fixed seeds; correct schema and splits.
- Estimation: on synthetic datasets, |β_hat − β_true| small with reasonable SE; likelihood concave and optimizer converges.
- Quality: tests pass; linters clean; configs documented; CLI runs end-to-end.

Risks and Mitigations

- Q scale drift affects β identifiability: fix reward/γ; use Q-snapshots from data time.
- Illegal actions: always restrict softmax to legal actions both in data and estimation.
- Sparse reward: start with step penalty + terminal bonus; tweak if learning stalls.
- Small sample bias: generate enough episodes; report uncertainty via SE/bootstrapping.

Minimal Directory Layout

- src/<pkg>/envs/peg/peg_env.py — Gym env with action mask.
- src/<pkg>/models/dqn.py — DQN/DoubleDQN.
- src/<pkg>/data/generate.py — rollout with β_true; write Parquet/JSONL + card.
- src/<pkg>/estimation/beta_mle.py — likelihood, gradient, Hessian, optimizer.
- src/<pkg>/inference/cli.py — subcommands: train, gen, estimate.
- configs/ — train_dqn.yaml, gen_data.yaml, estimate_beta.yaml.
- tests/ — test_peg_env.py, test_dqn_smoke.py, test_beta_mle.py.
- artifacts/ — models, datasets, metrics (created at runtime).

Key Config Knobs

- Train: gamma, lr, batch_size, target_update, replay_capacity, beta_train.
- Generate: beta_true, episodes, max_steps, seed, output_dir.
- Estimate: input_path, method (newton/brent), beta_bounds, tol, max_iter.

Suggested Next Steps

- Confirm environment variant (7x7 English board) and reward scale.
- Lock initial β_true set for data generation.
- Scaffold code skeletons, configs, and minimal tests (no heavy logic).
- Implement MLE on synthetic Q arrays and verify β recovery; then integrate with env data.
