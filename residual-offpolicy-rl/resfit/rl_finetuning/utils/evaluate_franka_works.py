# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np
import torch
from PIL import Image, ImageDraw
import torch.nn.functional as F

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
import torch.nn.functional as F

import wandb
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.scripts.residual_client import ResidualClient


def process_image_batch_dim(obs_dict, image_keys, out_size=84):
    """Convert image observations to batched float32 CHW tensors in [0, 1]."""
    imgs = []
    original_shapes = {}

    for k in image_keys:
        x = obs_dict[k]
        original_shapes[k] = x.shape

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

    # [N,3,H,W]
    imgs = torch.stack(imgs, dim=0)

    imgs = F.interpolate(
        imgs,
        size=(out_size, out_size),
        mode="bilinear",
        align_corners=False,
    )

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

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    title_suffix = f" (step={global_step})" if global_step is not None else ""
    ax3d.set_title(f"{title_prefix} xyz trajectories{title_suffix}")
    if ax3d.get_legend_handles_labels()[0]:
        ax3d.legend(loc="best", fontsize=8)

    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_title("xy projection"); ax_xy.grid(True, alpha=0.3)
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z"); ax_xz.set_title("xz projection"); ax_xz.grid(True, alpha=0.3)
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z"); ax_yz.set_title("yz projection"); ax_yz.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

def run_franka_evaluation(
    *,
    env: ResidualClient,
    agent: QAgent,
    num_episodes: int = 20,
    device: torch.device | str = "cuda",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = False,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
) -> tuple[dict[str, float], float]:
    """Extended evaluation to match the richer functionality available in
    the *residual_td3_dexmg* evaluator.  In particular, this version:

    1. Annotates every rendered frame with useful metadata (env index,
       episode counter, step counter, predicted Q-value and SUCCESS/FAIL).
    2. Caches frames per-episode and flushes them into a single video file
       at the end of the evaluation.
    3. Keeps the original simple success-rate / return metrics so existing
       training code continues to work unchanged.
    """


    device = torch.device(device)
    agent.eval()

    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1

    successes: list[bool] = []

    ep_combined_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_residual_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    all_combined_trajs: list[np.ndarray] = []
    all_residual_trajs: list[np.ndarray] = []

    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None

    done_episodes = 0
    obs = env.reset("ddd")

    progress_dots = ["."] * num_episodes
    print(f"Evaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)
    image_keys = [
        "observation.images.wrist_image_left",
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
    ]

    while done_episodes < num_episodes:
        with torch.no_grad():
            obs = process_image_batch_dim(obs, image_keys, out_size=84)
            actions = q_actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

        next_obs, reward, done, info = env.step(actions, eval=True)
        combined_action = info["combined_action"]
        residual_action = info["residual_action"]
        done_flags = done

        combined_xyz = _to_xyz_steps(combined_action)
        residual_xyz = _to_xyz_steps(residual_action)

        for env_idx in range(num_envs):
            ep_combined_buffers[env_idx].append(combined_xyz.copy())
            ep_residual_buffers[env_idx].append(residual_xyz.copy())

        if done_flags:
            is_success = bool(reward.item() > 0.9)
            print("rw: ", reward.item())
            progress_dots[done_episodes] = "✓" if is_success else "✗"
            print(f"\rEvaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

            successes.append(is_success)

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

            ep_combined_buffers[0] = []
            ep_residual_buffers[0] = []

            done_episodes += 1

        obs = next_obs

    print("Done")

    success_rate: float = float(np.mean(successes)) if successes else 0.0
    metrics: dict[str, float] = {"eval/success_rate": success_rate}

    fig_combined = _plot_xyz_trajectories(
        all_combined_trajs, successes,
        title_prefix="Combined action", global_step=global_step,
    )
    fig_residual = _plot_xyz_trajectories(
        all_residual_trajs, successes,
        title_prefix="Residual action", global_step=global_step,
    )

    if wandb.run is not None:
        wandb.log(
            {
                **metrics,
                "eval/xyz_trajectories_combined": wandb.Image(fig_combined),
                "eval/xyz_trajectories_residual": wandb.Image(fig_residual),
            },
            step=global_step,
        )

    plt.close(fig_combined)
    plt.close(fig_residual)

    agent.train(True)
    return metrics
