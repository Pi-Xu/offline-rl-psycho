from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from mdpmm.envs import make_env
from mdpmm.models.dqn import DQNAgent, ReplayBuffer, Transition
from mdpmm.utils.config import AppSettings, TrainDqnConfig, materialize_run_dir
from mdpmm.utils.io import append_jsonl, ensure_dir, save_json
from mdpmm.utils.logging import setup_logging
from mdpmm.utils.seeding import set_global_seeds
from mdpmm.training.start_sampler import ReverseStartSampler
import matplotlib
from io import BytesIO
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def _safe_import_matplotlib_pyplot():
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    f = max(0.0, min(1.0, 1.0 - step / float(decay_steps)))
    return end + (start - end) * f


def evaluate(env_id: str, agent: DQNAgent, episodes: int, max_steps: int, seed: int) -> Dict[str, float]:
    env = make_env(env_id)
    rng = np.random.RandomState(seed + 123)
    returns = []
    steps_list = []
    solved_count = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        total_r = 0.0
        steps = 0
        for t in range(max_steps):
            legal = info["action_mask"]
            action = agent.act(obs, legal_mask=legal, epsilon=0.0)
            sr = env.step(action)
            obs, info = sr.obs, {"action_mask": env.legal_action_mask()}
            total_r += sr.reward
            steps += 1
            if sr.done:
                if sr.info.get("solved"):
                    solved_count += 1
                break
        returns.append(total_r)
        steps_list.append(steps)
    success_rate = float(solved_count / episodes) if episodes > 0 else 0.0
    return {
        "success_rate": success_rate,
        "avg_return": float(np.mean(returns) if returns else 0.0),
        "avg_steps": float(np.mean(steps_list) if steps_list else 0.0),
    }


def _render_board(
    valid_mask: np.ndarray, board: np.ndarray, out_path: str, *, title: str = "Final Board"
) -> None:
    plt = _safe_import_matplotlib_pyplot()
    ensure_dir(os.path.dirname(out_path))
    # Build a color map: -1 invalid, 0 empty valid, 1 peg
    H, W = valid_mask.shape
    img = np.full((H, W), -1, dtype=np.int8)
    img[valid_mask.astype(bool)] = 0
    img[board.astype(bool)] = 1

    # Colors: invalid=light gray, empty=white, peg=steelblue
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["lightgray", "white", "steelblue"])  # indices: -1->0, 0->1, 1->2
    # Shift values to [0,2]
    img_disp = img + 1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_disp, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_board_frame(
    valid_mask: np.ndarray,
    board: np.ndarray,
    *,
    title: str,
    step: int,
    reward: float,
    cum_return: float,
) -> Image | None:
    plt = _safe_import_matplotlib_pyplot()
    # Build a color map: -1 invalid, 0 empty valid, 1 peg
    H, W = valid_mask.shape
    img = np.full((H, W), -1, dtype=np.int8)
    img[valid_mask.astype(bool)] = 0
    img[board.astype(bool)] = 1

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["lightgray", "white", "steelblue"])  # indices: -1->0, 0->1, 1->2
    img_disp = img + 1
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_disp, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Overlay text with reward info
    ax.text(
        0.02,
        0.96,
        f"Step {step} | r={reward:.2f} | R={cum_return:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    fig.tight_layout()

    if Image is None:
        plt.close(fig)
        return None
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    try:
        im = Image.open(buf).convert("RGB")
        return im
    finally:
        buf.close()


def train_dqn(config: TrainDqnConfig) -> None:
    logger = setup_logging(name="mdpmm.train")
    AppSettings()  # load .env defaults if present
    seed = set_global_seeds(config.seed)

    # If Hydra provided a run directory, prefer it to avoid duplicate nesting
    run_dir = config.run_dir or materialize_run_dir(config.artifacts_dir, config.run_id)
    ensure_dir(run_dir)
    save_json(json.loads(config.model_dump_json()), os.path.join(run_dir, "config.json"))

    env = make_env(config.env_id)
    obs_dim = int(np.prod(env.obs_shape))
    num_actions = env.num_actions
    agent = DQNAgent(
        obs_dim,
        num_actions,
        lr=config.lr,
        gamma=config.gamma,
        device=config.device,
        model_type=config.model_type,
        obs_shape=env.valid_mask.shape,
        cnn_channels=tuple(config.cnn_channels),
        cnn_hidden=int(config.cnn_hidden),
    )
    buf = ReplayBuffer(config.replay_capacity)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    reverse_sampler = None
    if config.use_reverse_starts:
        reverse_sampler = ReverseStartSampler.from_config(
            env,
            env_id=config.env_id,
            pool_dir=config.reverse_pool_dir,
            k_values=config.reverse_k_values,
            goal_root_spec=config.reverse_goal_roots,
            pool_size=config.reverse_pool_per_k,
            dedup_symmetry=config.reverse_dedup_symmetry,
            base_seed=config.reverse_seed,
            sampling_mode=config.reverse_sampling_mode,
            phase_len_episodes=config.reverse_phase_len_episodes,
            max_attempts=config.reverse_max_attempts,
        )
        logger.info(
            "Reverse starts enabled mode=%s ks=%s pool_dir=%s",
            config.reverse_sampling_mode,
            list(config.reverse_k_values),
            config.reverse_pool_dir,
        )
    # Track best by highest avg_return; tie-break by higher success_rate, then fewer avg_steps
    best = {"avg_return": float("-inf"), "success_rate": -1.0, "avg_steps": float("inf")}
    global_step = 0
    eval_stats = {}

    for ep in range(1, config.train_episodes + 1):
        if reverse_sampler is not None:
            board, reverse_meta = reverse_sampler.sample(ep - 1)
            obs, info = env.reset_to_board(board)
        else:
            obs, info = env.reset(seed=seed + ep)
            reverse_meta = None
        total_r = 0.0
        steps = 0
        for t in range(config.max_steps_per_episode):
            epsilon = linear_epsilon(global_step, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
            legal = info["action_mask"]
            action = agent.act(obs, legal_mask=legal, epsilon=epsilon)
            sr = env.step(action)
            next_obs = sr.obs
            done = sr.done
            next_legal = env.legal_action_mask()
            buf.add(
                Transition(
                    s=obs,
                    a=action,
                    r=sr.reward,
                    ns=next_obs,
                    done=done,
                    legal_mask_s=legal,
                    legal_mask_ns=next_legal,
                )
            )
            obs, info = next_obs, {"action_mask": next_legal}
            total_r += sr.reward
            steps += 1

            if len(buf) >= config.learning_starts and (global_step % 1 == 0):
                batch = buf.sample(config.batch_size)
                stats = agent.update(batch)
                if global_step % 100 == 0:
                    append_jsonl({"step": global_step, "kind": "train", **stats}, metrics_path)

            if global_step % config.target_update_interval == 0 and global_step > 0:
                agent.sync_target()

            global_step += 1
            if done:
                break

        # Per-episode log
        if config.print_episode and (ep % max(1, config.episode_log_interval) == 0):
            if reverse_meta is not None:
                logger.info(
                    "episode=%d return=%.3f steps=%d epsilon=%.3f start_k=%d start_root=%s",
                    ep,
                    total_r,
                    steps,
                    epsilon,
                    reverse_meta["k"],
                    reverse_meta["root"],
                )
            else:
                logger.info(
                    "episode=%d return=%.3f steps=%d epsilon=%.3f",
                    ep,
                    total_r,
                    steps,
                    epsilon,
                )
        append_jsonl(
            {
                "episode": ep,
                "kind": "episode",
                "return": total_r,
                "steps": steps,
                "epsilon": epsilon,
                **(
                    {
                        "start_k": reverse_meta["k"],
                        "start_root": reverse_meta["root"],
                        "start_pegs": reverse_meta["pegs"],
                    }
                    if reverse_meta is not None
                    else {}
                ),
            },
            metrics_path,
        )

        # Periodic evaluation
        if ep % config.eval_every == 0 or ep == config.train_episodes:
            eval_stats = evaluate(
                env_id=config.env_id,
                agent=agent,
                episodes=config.eval_episodes,
                max_steps=config.max_steps_per_episode,
                seed=seed + 999,
            )
            if config.print_eval:
                logger.info(
                    "eval@ep=%d success_rate=%.3f avg_steps=%.2f avg_return=%.3f",
                    ep,
                    eval_stats["success_rate"],
                    eval_stats["avg_steps"],
                    eval_stats["avg_return"],
                )
            append_jsonl({"episode": ep, "kind": "eval", **eval_stats}, metrics_path)

            # Save last checkpoint
            last_path = os.path.join(run_dir, "last.pt")
            agent.save(last_path, meta={"episode": ep, **eval_stats})

            # Track best by avg_return; tie-break by success_rate, then shorter avg_steps
            if eval_stats["avg_return"] > best["avg_return"]:
                is_better = True
            elif eval_stats["avg_return"] == best["avg_return"]:
                if eval_stats["success_rate"] > best["success_rate"]:
                    is_better = True
                elif eval_stats["success_rate"] == best["success_rate"]:
                    is_better = eval_stats["avg_steps"] < best["avg_steps"]
                else:
                    is_better = False
            else:
                is_better = False
            if is_better:
                best = {
                    "avg_return": eval_stats["avg_return"],
                    "success_rate": eval_stats["success_rate"],
                    "avg_steps": eval_stats["avg_steps"],
                }
                best_path = os.path.join(run_dir, "best.pt")
                agent.save(best_path, meta={"episode": ep, **eval_stats})

    # Write final summary
    save_json({"best": best, "last_eval": eval_stats}, os.path.join(run_dir, "summary.json"))

    # Greedy rollout with best checkpoint and optional rendering
    try:
        best_path = os.path.join(run_dir, "best.pt")
        if os.path.exists(best_path) and (config.render_final_image or config.render_gif):
            env_vis = make_env(config.env_id)
            obs_dim_vis = int(np.prod(env_vis.obs_shape))
            agent_vis, meta = DQNAgent.from_checkpoint(
                best_path,
                obs_dim=obs_dim_vis,
                num_actions=env_vis.num_actions,
                lr=config.lr,
                gamma=config.gamma,
                device=config.device,
                model_type=config.model_type,
                obs_shape=env_vis.valid_mask.shape,
                cnn_channels=tuple(config.cnn_channels),
                cnn_hidden=int(config.cnn_hidden),
            )
            obs, info = env_vis.reset(seed=seed + 2024)
            steps: int = 0
            solved_flag: bool = False
            frames: List[Image] = [] if (Image is not None and config.render_gif) else []
            cum_ret: float = 0.0
            # Initial frame (before any move)
            if Image is not None and config.render_gif:
                board0 = obs.reshape(env_vis.valid_mask.shape)
                f0 = _render_board_frame(
                    env_vis.valid_mask,
                    board0,
                    title="Best Model — Rollout",
                    step=steps,
                    reward=0.0,
                    cum_return=0.0,
                )
                if f0 is not None:
                    frames.append(f0)
            for _ in range(config.max_steps_per_episode):
                a = agent_vis.act(obs, legal_mask=info["action_mask"], epsilon=0.0)
                sr = env_vis.step(a)
                obs, info = sr.obs, {"action_mask": env_vis.legal_action_mask()}
                steps += 1
                cum_ret += float(sr.reward)
                if Image is not None and config.render_gif:
                    board_t = obs.reshape(env_vis.valid_mask.shape)
                    ft = _render_board_frame(
                        env_vis.valid_mask,
                        board_t,
                        title="Best Model — Rollout",
                        step=steps,
                        reward=float(sr.reward),
                        cum_return=cum_ret,
                    )
                    if ft is not None:
                        frames.append(ft)
                if sr.done:
                    solved_flag = bool(sr.info.get("solved", False))
                    break
            # Reconstruct board from observation
            HxW = env_vis.obs_shape[0]
            H = W = int(np.sqrt(HxW)) if int(np.sqrt(HxW)) ** 2 == HxW else None
            if H is None:
                # Use valid_mask shape if non-square
                valid_mask = env_vis.valid_mask
                board = obs.reshape(valid_mask.shape)
            else:
                valid_mask = env_vis.valid_mask
                board = obs.reshape(valid_mask.shape)
            if config.render_final_image:
                out_img = os.path.join(run_dir, "best_inference.png")
                _render_board(valid_mask, board, out_img, title="Best Model — Final Board")
            else:
                out_img = ""
            # Save a small JSON record
            save_json(
                {
                    "steps": steps,
                    "solved": solved_flag,
                    "meta": meta,
                    "checkpoint": os.path.basename(best_path),
                    "image": os.path.basename(out_img),
                },
                os.path.join(run_dir, "best_inference.json"),
            )
            # Save GIF video if possible
            if config.render_gif and Image is not None and frames:
                gif_path = os.path.join(run_dir, "best_inference.gif")
                try:
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=max(10, int(config.gif_duration_ms)),  # ms per frame
                        loop=0,
                        optimize=True,
                    )
                except Exception:
                    pass
    except Exception:
        # Rendering should not break training runs
        pass
