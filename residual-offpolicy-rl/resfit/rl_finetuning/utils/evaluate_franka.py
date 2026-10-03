# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio
import matplotlib
matplotlib.use("Agg")   # Select the backend before importing pyplot.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import wandb
from resfit.rl_finetuning.off_policy.rl.q_agent_lang import QAgentLang
from resfit.rl_finetuning.scripts.gpt_residual_robot import BasePolicy
from loguru import logger
import sys

def process_image_batch_dim(obs_dict, image_keys, out_size=84):
    """Convert image observations to batched float32 CHW tensors in [0, 1]."""
    imgs = []
    for k in image_keys:
        x = obs_dict[k]
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 3:
            raise ValueError(f"{k} expected 3 dims [H,W,C], got shape={x.shape}")

        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        elif x.shape[0] == 3:
            pass
        else:
            raise ValueError(f"{k} is neither HWC nor CHW, got shape={x.shape}")

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        imgs.append(x)

    imgs = torch.stack(imgs, dim=0)
    imgs = F.interpolate(imgs, size=(out_size, out_size), mode="bilinear", align_corners=False)

    for i, k in enumerate(image_keys):
        obs_dict[k] = imgs[i].contiguous().unsqueeze(0)
    return obs_dict


def _to_xyz_steps(arr) -> np.ndarray:
    """Extract XYZ actions as a (T, 3) array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    if arr.ndim == 3:        # [B, T, A] -> batch 0
        return arr[0, :, :3]
    elif arr.ndim == 2:      # [T, A]
        return arr[:, :3]
    elif arr.ndim == 1:      # [A]
        return arr[None, :3]
    else:
        return arr.reshape(-1)[None, :3]

def setup_logger(log_dir: str | Path):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "log.txt"

        logger.remove()
        logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def _plot_residual_per_dim(
    trajectories: list[np.ndarray],
    successes: list[bool],
    title_prefix: str = "Residual action",
    global_step: int | None = None,
):
    """Plot residual XYZ actions over episode steps, colored by success."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    dim_names = ["x", "y", "z"]

    drew_success_label = False
    drew_fail_label = False
    for traj, ok in zip(trajectories, successes):
        if traj.shape[0] == 0:
            continue
        steps = np.arange(traj.shape[0])
        color = "tab:green" if ok else "tab:red"

        for d in range(3):
            label = None
            if ok and not drew_success_label and d == 0:
                label = "success"
            elif (not ok) and not drew_fail_label and d == 0:
                label = "fail"
            axes[d].plot(
                steps, traj[:, d],
                color=color, alpha=0.6, linewidth=1.0, label=label,
            )
        if ok:
            drew_success_label = True
        else:
            drew_fail_label = True

    title_suffix = f" (step={global_step})" if global_step is not None else ""
    for d, name in enumerate(dim_names):
        axes[d].set_ylabel(f"residual {name}")
        axes[d].grid(True, alpha=0.3)
        axes[d].axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
    axes[0].set_title(f"{title_prefix} per-dim over steps{title_suffix}")
    axes[-1].set_xlabel("episode step")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def _plot_xyz_trajectories(
    trajectories: list[np.ndarray],
    successes: list[bool],
    title_prefix: str = "Combined action",
    global_step: int | None = None,
):
    """Plot XYZ trajectories and their projections, colored by episode success."""
    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    for i, (traj, ok) in enumerate(zip(trajectories, successes)):
        if traj.shape[0] == 0:
            continue
        color = "tab:green" if ok else "tab:red"
        alpha = 0.75
        label = None
        if ok and not any(s for s in successes[:i]):
            label = "success"
        elif (not ok) and not any((not s) for s in successes[:i]):
            label = "fail"

        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        ax3d.plot(x, y, z, color=color, alpha=alpha, linewidth=1.2, label=label)
        ax3d.scatter(x[0], y[0], z[0], color=color, marker="o", s=20)
        ax3d.scatter(x[-1], y[-1], z[-1], color=color, marker="x", s=30)

        ax_xy.plot(x, y, color=color, alpha=alpha, linewidth=1.0)
        ax_xz.plot(x, z, color=color, alpha=alpha, linewidth=1.0)
        ax_yz.plot(y, z, color=color, alpha=alpha, linewidth=1.0)

    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    title_suffix = f" (step={global_step})" if global_step is not None else ""
    ax3d.set_title(f"{title_prefix} xyz trajectories{title_suffix}")
    if ax3d.get_legend_handles_labels()[0]:
        ax3d.legend(loc="best", fontsize=8)

    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_title("xy projection"); ax_xy.grid(True, alpha=0.3)
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z"); ax_xz.set_title("xz projection"); ax_xz.grid(True, alpha=0.3)
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z"); ax_yz.set_title("yz projection"); ax_yz.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_q_trajectories(
    trajectories: list[list[float]],
    successes: list[bool],
    global_step: int | None = None,
):
    """Plot per-episode Q-values and their distributions at normalized episode steps."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    drew_success_label = False
    drew_fail_label = False
    for traj, ok in zip(trajectories, successes):
        if not traj:
            continue
        steps = list(range(len(traj)))
        if ok:
            label = None if drew_success_label else "Success"
            drew_success_label = True
            ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label=label)
        else:
            label = None if drew_fail_label else "Failure"
            drew_fail_label = True
            ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label=label)

    title_suffix = f" (step={global_step})" if global_step is not None else ""
    ax1.set_xlabel("Episode Step")
    ax1.set_ylabel("Q-Value")
    ax1.set_title(f"Q-Value Trajectories Over Time{title_suffix}")
    ax1.grid(True, alpha=0.3)
    if drew_success_label or drew_fail_label:
        ax1.legend()

    progress_points = [0.25, 0.5, 0.75, 1.0]
    bucket: dict[str, list[float]] = {f"{int(p * 100)}%": [] for p in progress_points}
    for traj in trajectories:
        if not traj:
            continue
        L = len(traj)
        for p in progress_points:
            idx = min(int(p * L), L - 1)
            bucket[f"{int(p * 100)}%"].append(traj[idx])

    box_data = [bucket[f"{int(p * 100)}%"] for p in progress_points]
    box_labels = [f"{int(p * 100)}%" for p in progress_points]
    ax2.boxplot(box_data, labels=box_labels)
    ax2.set_xlabel("Episode Progress")
    ax2.set_ylabel("Q-Value")
    ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _inject_task_emb(
    obs: dict[str, torch.Tensor], task_emb: torch.Tensor, key: str = "observation.task_emb"
) -> dict[str, torch.Tensor]:
    """Set ``obs[key]`` to ``task_emb`` (broadcast to whatever batch dim the
    rest of obs has).  Mutates ``obs`` in place and returns it for chaining.
    """
    if key in obs:
        return obs
    state = obs.get("observation.state", None)
    if state is not None and state.dim() == 2:
        emb = task_emb.unsqueeze(0).expand(state.size(0), -1).contiguous()
    else:
        emb = task_emb
    obs[key] = emb.to(state.device if state is not None else task_emb.device)
    return obs

def _attach_task_context(obs, agent, lang_cfg, task_emb, task_id: int):
    if obs is None:
        return obs
    if lang_cfg.enabled:
        if task_emb is None:
            raise RuntimeError("task_emb is required when language conditioning is enabled")
        _inject_task_emb(obs, task_emb, key=lang_cfg.lang_emb_obs_key)
    if getattr(agent, "task_specific_actor", False):
        state = obs.get("observation.state")
        device = state.device if isinstance(state, torch.Tensor) else None
        if isinstance(state, torch.Tensor) and state.ndim == 2:
            value = torch.full(
                (state.shape[0],), int(task_id), dtype=torch.long, device=device
            )
        else:
            value = torch.tensor(int(task_id), dtype=torch.long, device=device)
        obs[agent.task_id_obs_key] = value
    return obs


def _save_eval_progress(
    path: Path,
    *,
    eval_step: int,
    eval_num_episode: int,
    chunk_len: int,
    candidate_tasks: list[str],
    successes_by_task: dict[str, list[bool]],
    all_combined_trajs: list[np.ndarray],
    all_residual_trajs: list[np.ndarray],
    all_q_trajectories_by_task: dict[str, list[list[float]]],
    status: str,
) -> None:
    """Atomically persist completed evaluation episodes as portable JSON."""
    payload = {
        "schema_version": 1,
        "status": status,
        "eval_step": int(eval_step),
        "eval_num_episode": int(eval_num_episode),
        "chunk_len": int(chunk_len),
        "candidate_tasks": candidate_tasks,
        "successes_by_task": successes_by_task,
        "all_combined_trajs": [trajectory.tolist() for trajectory in all_combined_trajs],
        "all_residual_trajs": [trajectory.tolist() for trajectory in all_residual_trajs],
        "all_q_trajectories_by_task": all_q_trajectories_by_task,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _load_eval_progress(
    path: Path,
    *,
    eval_step: int,
    eval_num_episode: int,
    chunk_len: int,
    candidate_tasks: list[str],
) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "schema_version": 1,
        "eval_step": int(eval_step),
        "eval_num_episode": int(eval_num_episode),
        "chunk_len": int(chunk_len),
        "candidate_tasks": candidate_tasks,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Evaluation progress at {path} is incompatible with this run: {mismatches}. "
            "Move or delete that file to start this evaluation from scratch."
        )
    return payload


def _save_eval_agent_snapshot(
    path: Path,
    *,
    agent: QAgentLang,
    eval_step: int,
    chunk_len: int,
    candidate_tasks: list[str],
) -> None:
    payload = {
        "schema_version": 1,
        "eval_step": int(eval_step),
        "chunk_len": int(chunk_len),
        "candidate_tasks": candidate_tasks,
        "agent_state": {
            key: value.detach().cpu()
            for key, value in agent.state_dict().items()
        },
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _load_eval_agent_snapshot(
    path: Path,
    *,
    agent: QAgentLang,
    device: torch.device,
    eval_step: int,
    chunk_len: int,
    candidate_tasks: list[str],
) -> None:
    snapshot = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "schema_version": 1,
        "eval_step": int(eval_step),
        "chunk_len": int(chunk_len),
        "candidate_tasks": candidate_tasks,
    }
    mismatches = {
        key: (snapshot.get(key), value)
        for key, value in expected.items()
        if snapshot.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Evaluation agent snapshot at {path} is incompatible: {mismatches}"
        )
    agent.load_state_dict(snapshot["agent_state"], strict=True)

# Main eval loop
def run_franka_evaluation(
    *,
    env: BasePolicy,
    agent: QAgentLang,
    eval_num_episode: int = 20,
    device: torch.device | str = "cuda",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = True,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
    lang_cfg,
    lang_embedder,
    chunk_len: int = 1,
    resume_progress: bool = True,
    initialize_task_success_metrics: bool = False,
) -> dict[str, Any]:
    device = torch.device(device)
    agent.eval()
    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1
    eval_step = global_step or 0
    cycle_dir = env.task_reward_generator.set_output_cycle(eval_step)
    setup_logger(cycle_dir)
    logger.info(f"---------------- evaluation after {eval_step} steps ----------------")
    candidate_tasks = list(env.task_reward_generator.candidate_tasks)
    if not candidate_tasks:
        raise ValueError("env.task_reward_generator.candidate_tasks is empty")
    logger.info(f"Evaluating each task for {eval_num_episode} episodes:")
    for index, task in enumerate(candidate_tasks, start=1):
        logger.info(f"task {index}: {task}")

    progress_path = Path(cycle_dir) / "evaluation_progress.json"
    if not resume_progress:
        progress_path.unlink(missing_ok=True)
    saved_progress = (
        _load_eval_progress(
            progress_path,
            eval_step=eval_step,
            eval_num_episode=eval_num_episode,
            chunk_len=chunk_len,
            candidate_tasks=candidate_tasks,
        )
        if resume_progress
        else None
    )
    snapshot_path = Path(cycle_dir) / "evaluation_agent.pt"
    evaluation_incomplete = (
        saved_progress is None or saved_progress.get("status") != "complete"
    )
    if evaluation_incomplete:
        if resume_progress and snapshot_path.exists():
            _load_eval_agent_snapshot(
                snapshot_path,
                agent=agent,
                device=device,
                eval_step=eval_step,
                chunk_len=chunk_len,
                candidate_tasks=candidate_tasks,
            )
            logger.info(f"Restored evaluation agent snapshot from {snapshot_path}")
        elif saved_progress is not None:
            raise RuntimeError(
                f"Evaluation progress exists at {progress_path}, but its agent "
                f"snapshot is missing at {snapshot_path}. Refusing to mix policies."
            )
        else:
            _save_eval_agent_snapshot(
                snapshot_path,
                agent=agent,
                eval_step=eval_step,
                chunk_len=chunk_len,
                candidate_tasks=candidate_tasks,
            )
            logger.info(f"Saved evaluation agent snapshot to {snapshot_path}")

    successes_by_task: dict[str, list[bool]] = {
        task: list(saved_progress["successes_by_task"].get(task, []))
        if saved_progress is not None else []
        for task in candidate_tasks
    }
    successes: list[bool] = [
        success
        for task in candidate_tasks
        for success in successes_by_task[task]
    ]
    ep_combined_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_residual_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_q_preds: list[list[float]] = [[] for _ in range(num_envs)]

    all_combined_trajs: list[np.ndarray] = (
        [np.asarray(trajectory, dtype=np.float32) for trajectory in saved_progress["all_combined_trajs"]]
        if saved_progress is not None else []
    )
    all_residual_trajs: list[np.ndarray] = (
        [np.asarray(trajectory, dtype=np.float32) for trajectory in saved_progress["all_residual_trajs"]]
        if saved_progress is not None else []
    )
    all_q_trajectories_by_task: dict[str, list[list[float]]] = {
        task: list(saved_progress["all_q_trajectories_by_task"].get(task, []))
        if saved_progress is not None else []
        for task in candidate_tasks
    }

    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None

    done_episodes = len(successes)
    total_episodes = eval_num_episode * len(candidate_tasks)
    if not (
        done_episodes == len(all_combined_trajs) == len(all_residual_trajs)
        and all(len(successes_by_task[task]) <= eval_num_episode for task in candidate_tasks)
        and all(
            len(all_q_trajectories_by_task[task]) == len(successes_by_task[task])
            for task in candidate_tasks
        )
    ):
        raise RuntimeError(f"Inconsistent evaluation progress in {progress_path}")
    if done_episodes > total_episodes:
        raise RuntimeError(
            f"Evaluation progress has {done_episodes} episodes, expected at most {total_episodes}"
        )

    task_index = min(done_episodes // eval_num_episode, len(candidate_tasks) - 1)
    task_prompt = candidate_tasks[task_index]
    logger.info(f"---------------- task {task_index + 1}: {task_prompt} ----------------")
    env.evaluation = True

    progress_by_task = {
        task: [
            *("✓" if success else "✗" for success in successes_by_task[task]),
            *(["."] * (eval_num_episode - len(successes_by_task[task]))),
        ]
        for task in candidate_tasks
    }
    if saved_progress is not None:
        logger.info(
            f"Resuming evaluation from {progress_path}: "
            f"{done_episodes}/{total_episodes} episodes already complete"
        )
        # Keep artifact numbering monotonic after a process restart so images
        # belonging to completed episodes are not overwritten.
        env.task_reward_generator.round = max(
            int(env.task_reward_generator.round), done_episodes
        )
        if hasattr(env, "round"):
            env.round = max(int(env.round), done_episodes)

    if done_episodes < total_episodes:
        obs = env.reset(task_prompt, evaluate_previous=False)
        task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
        obs = _attach_task_context(obs, agent, lang_cfg, task_emb, task_index)
        logger.info(
            f"Evaluating {eval_num_episode} episodes: {''.join(progress_by_task[task_prompt])}",
            end="",
            flush=True,
        )
    image_keys = [
        "observation.images.wrist_image_left",
        "observation.images.exterior_image_2_left",
    ]

    while done_episodes < total_episodes:
        # 1. Policy + Q prediction
        with torch.no_grad():
            obs = process_image_batch_dim(obs, image_keys, out_size=84)
            # Attach the base-action chunk (H*dim) so actor/critic see chunk-level base.
            obs["observation.base_action"] = env.current_base_chunk(chunk_len)
            actions = q_actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs_q = agent._augment_state(obs, detach_lang=True)
            # On-the-fly features for Q-value prediction
            obs_q = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs_q.items()}
            obs_q["feat"] = agent._encode(obs_q, augment=False)

            if getattr(agent, "residual_actor", False) and "observation.base_action" in obs:
                q_actions = torch.clamp(obs["observation.base_action"] + actions, -1.0, 1.0)

            q_pred = agent.predict_q(obs_q, q_actions).detach().cpu().squeeze(-1)
            q_pred = q_pred.reshape(-1)

        # 2. Env step
        # Open-loop execute the whole predicted residual chunk.
        next_obs, combined_chunk, reward, done, info = env.step_chunk(
            actions, task_prompt=task_prompt, evaluation=True
        )

        combined_action = info["combined_action"]
        residual_action = info["residual_action"]
        done_flags = done

        combined_xyz = _to_xyz_steps(combined_action)
        residual_xyz = _to_xyz_steps(residual_action)

        for env_idx in range(num_envs):
            ep_combined_buffers[env_idx].append(combined_xyz.copy())
            ep_residual_buffers[env_idx].append(residual_xyz.copy())
            q_val = q_pred[env_idx].item() if q_pred.numel() > env_idx else float(q_pred.mean().item())
            ep_q_preds[env_idx].append(q_val)

        if done_flags:
            is_success = bool(reward.item() > 0.9)           
            task_episode_index = len(successes_by_task[task_prompt])
            progress_by_task[task_prompt][task_episode_index] = "✓" if is_success else "✗"
            logger.info(f"{task_prompt}: {progress_by_task[task_prompt][task_episode_index]}")
            logger.info(
                f"Evaluating {eval_num_episode} episodes: "
                f"{''.join(progress_by_task[task_prompt])}",
                end="",
                flush=True,
            )
            successes_by_task[task_prompt].append(is_success)
            successes.append(is_success)
            if len(successes_by_task[task_prompt]) == eval_num_episode:
                task_successes = successes_by_task[task_prompt]
                success_count = sum(task_successes)
                success_rate = success_count / eval_num_episode
                logger.info(
                    f"task {task_index + 1} success rate: {success_rate:.2%} "
                    f"({success_count}/{eval_num_episode})"
                )
            combined_traj = (
                np.concatenate(ep_combined_buffers[0], axis=0)
                if ep_combined_buffers[0] else np.zeros((0, 3))
            )
            residual_traj = (
                np.concatenate(ep_residual_buffers[0], axis=0)
                if ep_residual_buffers[0] else np.zeros((0, 3))
            )
            all_combined_trajs.append(combined_traj)
            all_residual_trajs.append(residual_traj)

            all_q_trajectories_by_task[task_prompt].append(ep_q_preds[0].copy())

            ep_combined_buffers[0] = []
            ep_residual_buffers[0] = []

            ep_q_preds[0] = []

            done_episodes += 1
            _save_eval_progress(
                progress_path,
                eval_step=eval_step,
                eval_num_episode=eval_num_episode,
                chunk_len=chunk_len,
                candidate_tasks=candidate_tasks,
                successes_by_task=successes_by_task,
                all_combined_trajs=all_combined_trajs,
                all_residual_trajs=all_residual_trajs,
                all_q_trajectories_by_task=all_q_trajectories_by_task,
                status="in_progress",
            )
            logger.info(
                f"Saved evaluation progress: {done_episodes}/{total_episodes}"
            )
            logger.info("-------------------------------------------------------------------")

            next_task_index = min(done_episodes // eval_num_episode, len(candidate_tasks) - 1)
            if done_episodes < total_episodes and next_task_index != task_index:
                task_index = next_task_index
                task_prompt = candidate_tasks[task_index]
                logger.info(f"---------------- task {task_index + 1}: {task_prompt} ----------------")
                next_obs = env.reset(task_prompt, evaluate_previous=False)
                logger.info(
                    f"Evaluating {eval_num_episode} episodes: "
                    f"{''.join(progress_by_task[task_prompt])}",
                    end="",
                    flush=True,
                )

            task_emb = lang_embedder(task_prompt) if lang_embedder is not None else None
        next_obs = _attach_task_context(
            next_obs, agent, lang_cfg, task_emb, task_index
        )
        obs = next_obs

    logger.info("Done")
    env.evaluation = False

    # 5. Aggregate metrics + Q stats
    metrics: dict[str, float] = {}
    for index, task in enumerate(candidate_tasks):
        task_successes = successes_by_task[task]
        task_q_trajectories = all_q_trajectories_by_task[task]
        nonempty_q = [
            np.asarray(trajectory, dtype=np.float32)
            for trajectory in task_q_trajectories
            if trajectory
        ]
        flat_q = np.concatenate(nonempty_q) if nonempty_q else np.zeros((0,), dtype=np.float32)
        metric_prefix = f"eval/task_{index + 1}"
        metrics[f"{metric_prefix}/success_rate"] = (
            float(np.mean(task_successes)) if task_successes else 0.0
        )
        metrics[f"{metric_prefix}/q_mean"] = (
            float(flat_q.mean()) if flat_q.size else 0.0
        )

    # 6. Plots -> wandb
    fig_combined = _plot_xyz_trajectories(
        all_combined_trajs, successes,
        title_prefix="Combined action", global_step=global_step,
    )
    fig_residual = _plot_residual_per_dim(
        all_residual_trajs, successes,
        title_prefix="Residual action", global_step=global_step,
    )
    q_figures = {
        task: _plot_q_trajectories(
            all_q_trajectories_by_task[task],
            successes_by_task[task],
            global_step=global_step,
        )
        for task in candidate_tasks
    } if save_q_plots else {}

    if wandb.run is not None:
        log_dict = {
            **metrics,
            "eval/xyz_trajectories_combined": wandb.Image(fig_combined),
            "eval/xyz_trajectories_residual": wandb.Image(fig_residual),
        }
        if initialize_task_success_metrics:
            initial_rates = {
                task: metrics[f"eval/task_{index + 1}/success_rate"]
                for index, task in enumerate(candidate_tasks)
            }
            sampling_weights = {
                task: max(1e-6, 1.0 - initial_rates[task])
                for task in candidate_tasks
            }
            weight_total = sum(sampling_weights.values())
            for index, task in enumerate(candidate_tasks):
                log_dict[f"tasks/task_{index}_success_rate"] = initial_rates[task]
                log_dict[f"tasks/task_{index}_episode_count"] = 0
                log_dict[f"tasks/task_{index}_probability"] = (
                    sampling_weights[task] / weight_total
                )
                log_dict[f"tasks/task_{index}_attempts"] = 0
        for index, task in enumerate(candidate_tasks):
            if task in q_figures:
                log_dict[f"eval/task_{index + 1}/q_trajectories"] = wandb.Image(q_figures[task])
        wandb.log(log_dict, step=global_step)

    plt.close(fig_combined)
    plt.close(fig_residual)
    for figure in q_figures.values():
        plt.close(figure)

    _save_eval_progress(
        progress_path,
        eval_step=eval_step,
        eval_num_episode=eval_num_episode,
        chunk_len=chunk_len,
        candidate_tasks=candidate_tasks,
        successes_by_task=successes_by_task,
        all_combined_trajs=all_combined_trajs,
        all_residual_trajs=all_residual_trajs,
        all_q_trajectories_by_task=all_q_trajectories_by_task,
        status="complete",
    )
    snapshot_path.unlink(missing_ok=True)

    # Keep ordered per-episode outcomes out of the W&B metric payload, but
    # return them to the trainer so the initial evaluation itself becomes the
    # first full success-rate rolling window.
    metrics["_eval_outcomes_by_task"] = {
        task: [int(success) for success in successes_by_task[task]]
        for task in candidate_tasks
    }

    logger.info("--------------------------------------------------------------------------------------")

    agent.train(True)
    return metrics
