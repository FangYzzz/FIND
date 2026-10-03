# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path

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


    successes: list[bool] = []  # episode-level success flags


    # Video buffers -----------------------------------------------------
    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None

    done_episodes = 0
    obs = env.reset()

    # Initialize progress display with dots
    progress_dots = ["."] * num_episodes
    print(f"Evaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)
    image_keys=["observation.images.exterior_image_1_left",  # observation.images.exterior_image_1_left
                "observation.images.exterior_image_2_left",
                "observation.images.wrist_image_left",]
    while done_episodes < num_episodes:
        # 1. Policy inference + Q-value prediction ---------------------
        with torch.no_grad():
            obs =process_image_batch_dim(obs, image_keys, out_size=84)
            actions = q_actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # Build features on-the-fly to obtain Q-predictions --------

            # For Q-value computation, use combined action and clamp to [-1, 1] (consistent with training)
            if agent.residual_actor:
                q_actions = torch.clamp(obs["observation.base_action"] + actions, -1.0, 1.0)

        # 2. Environment step ------------------------------------------
        next_obs, reward, done, info = env.step(actions)
        done_flags = done

        # Capture frames ------------------------------------------------

        # 3. Per-environment bookkeeping -------------------------------


        if done_flags:
            # Episode finished -- aggregate results ----------------
            is_success = bool(reward.item() == 1.0)

            # Update progress display
            progress_dots[done_episodes] = "✓" if is_success else "✗"
            print(f"\rEvaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

            successes.append(is_success)

            # Store Q-trajectory data for plotting ------------------

            done_episodes += 1


        # Prepare for next loop ----------------------------------------
        obs = next_obs

    print("Done")

    # 4. Aggregate metrics ---------------------------------------------
    # Sanity check: episode lengths must align 1:1 with successes


    success_rate: float = float(np.mean(successes)) if successes else 0.0


    metrics: dict[str, float] = {
        "eval/success_rate": success_rate,
    }

    if wandb.run is not None:
        wandb.log(metrics, step=global_step)


    # Restore training mode --------------------------------------------
    agent.train(True)

    return metrics
