# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig, RLPDAlgoConfig, RLPDDexmgConfig


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: int | None = 300
    horizon: int = 400
    # Offline data action labeling options
    use_base_policy_for_base_actions: bool = True
    # Normalization safeguards
    min_action_range: float = 1e-1  # Minimum range for any action dimension to prevent normalization blow-up
    min_state_std: float = 1e-1  # Minimum std for any state dimension to prevent normalization blow-up


@dataclass
class WandBConfig:
    project: str = "robomimic-can-residual-td3"
    mode: str = "online"
    entity: str | None = None
    notes: str | None = None
    continue_run_id: str | None = None
    resume_checkpoint_run: bool = True  # Reuse the checkpoint W&B run when True.
    name: str | None = None
    group: str | None = None


@dataclass
class BasePolicyConfig:
    wandb_id: str = "dexmg-bc/o2h7mdwe"
    wt_type: str = "best"
    wt_version: str = "latest"


@dataclass
class RLTokenConfig:
    """RL-Token bottleneck and cache settings (disabled for legacy trainers)."""

    enabled: bool = False
    # Optional smoke-test limit. When set, this overrides
    # offline_data.num_episodes for embedding collection, VAE training, and
    # offline replay construction so every stage uses the same subset.
    offline_num_episodes: int | None = 590
    obs_key: str = "observation.rl_token"
    vla_embedding_key: str = "vla_embedding"
    embedding_cache_dir: str = "vla_embedding_cache"
    checkpoint_dir: str = "rl_token_checkpoints_10070_step65000"
    force_recollect_embeddings: bool = False
    # Number of embeddings buffered in RAM before an atomic disk shard is
    # written. Raw VLA sequences are never accumulated for the full dataset.
    embedding_shard_size: int = 32
    embedding_storage_dtype: str = "int8"
    force_retrain: bool = False
    token_dim: int = 512
    model_dim: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 4
    # Upper bound for padded sequence tokens per VAE minibatch. This
    # automatically reduces batch_size for long pi0 prefix sequences.
    max_tokens_per_batch: int = 1024
    epochs: int = 50
    validation_fraction: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    split_seed: int = 0
    max_train_samples_per_epoch: int | None = 10_000
    max_validation_samples: int | None = 2_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    beta_kl: float = 1e-4
    num_workers: int = 0
    # Request final VLA prefix features (all camera views + language tokens).
    request_field: str = "return_vla_embedding"


@dataclass
class ResidualTD3AlgoConfig(RLPDAlgoConfig):
    # Number of critic-only updates before training the actor
    critic_warmup_steps: int = 10_000

    # Scale for random action noise during initial exploration phase
    # Actions are sampled as: rand_actions = torch.rand(...) * 2 * random_action_noise_scale - random_action_noise_scale
    random_action_noise_scale: float = 0.2

    # Whether to use base policy + noise (True) or pure uniform noise (False) during warmup
    # Note: Environment wrapper always applies base_action + residual_action
    # True: residual_action = noise (resulting in base_action + noise)
    # False: residual_action = pure_random - base_action (resulting in pure_random)
    use_base_policy_for_warmup: bool = True

    # Whether a resumed training process should discard the per-task rolling
    # success-rate state and restart every task from the default 0.5 baseline.
    # Set to False to restore that state from the training checkpoint.
    reset_task_success_rates_on_restart: bool = False

    stddev_max: float = 0.003
    stddev_min: float = 0.003
    stddev_step: int = 100_000

    # Progressive clipping schedule for the residual actions
    # I.e., starts clipping linearly from 0 to action scale over progressive_clipping_steps steps
    progressive_clipping_steps: int = 0


@dataclass
class ResidualTD3DexmgConfig(RLPDDexmgConfig):
    actor_name: str | None = None  # Inferred from base policy config

    # The chunked multi-task trainer maps candidate_tasks[i] to actor_i. The
    # visual/RL-token encoder and task-conditioned critic remain shared.
    task_specific_actor: bool = True

    # Optionally append task/reward logs and artifacts to an existing run.
    # Training stays in the current step_<N>/log.txt; each evaluation switches
    # the logger to the directory for its own global step.
    task_output_root: str | None = None

    # Restrict online task sampling/evaluation to these exact task prompts.
    # None preserves the TaskRewardGenerator's current default task set.
    candidate_tasks: list[str] | None = None

    # Per-task success is a rolling mean over N binary outcomes. The initial
    # evaluation's ordered N outcomes seed the window; training episodes then
    # replace those evaluation outcomes one by one.
    task_success_window_size: int = 20

    # A task whose rolling success rate reaches 1.0 keeps this exact sampling
    # probability whenever at least one other feasible task is not mastered.
    # The remaining probability mass is distributed among the other feasible
    # tasks in proportion to 1 - success_rate.
    min_task_sample_probability: float = 0.05

    # Curriculum switch. True prioritizes feasible tasks with lower rolling
    # success rates; False samples uniformly from the feasible task set.
    prioritize_low_success_tasks: bool = True

    # A fresh run can reuse the ordered step-0 evaluation outcomes in this log
    # instead of evaluating the robot again. Set this and the overrides below
    # to None to run eval_first.
    # Checkpoint resume always restores the checkpoint's newer window state.
    initial_task_success_log: str | None = None

    # Optional per-task overrides applied on top of initial_task_success_log.
    # Each value must contain exactly task_success_window_size ordered 0/1
    # outcomes. Keys may be a subset of candidate_tasks when a log supplies the
    # other tasks; without a log, every candidate task must be provided.
    # These overrides apply only to fresh runs, never checkpoint resumes.
    initial_task_success_window_overrides: dict[str, list[int]] | None = None

    algo: ResidualTD3AlgoConfig = field(default_factory=ResidualTD3AlgoConfig)

    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,  # imp for residual
            ),
        )
    )

    offline_data: OfflineDataConfig | None = field(default_factory=OfflineDataConfig)

    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)

    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)

    wandb: WandBConfig = field(default_factory=WandBConfig)

    # Save robot debug-camera frames and GPT before/after scene images for
    # evaluation, online training, and online replay-buffer warmup.
    save_images: bool = True
    # Save one complete XYZ base/residual/combined action plot and its raw CSV
    # after every online training episode. This is independent of save_images.
    save_episode_action_plots: bool = False

    # Master switch for both the step-0 and periodic robot evaluations.
    eval_enabled: bool = True

    eval_interval_every_steps: int = 100_000

    # Whether to run an evaluation pass before training begins (at step 0).
    # This is bypassed when an initial log or window overrides are configured.
    eval_first: bool = True

    resume: bool = False
    resume_checkpoint: str | None = None
    # Set this only when the periodic evaluation at resume_checkpoint's
    # global_step completed before the previous process stopped. The collector
    # continues at the next chunk instead of repeating that step/evaluation;
    # later evaluation intervals are unchanged.
    skip_completed_eval_on_resume: bool = False
    # Collector checkpoint threshold. When it is crossed in the middle of an
    # episode, collection finishes that episode, pauses, and waits for the
    # learner to process the same step before saving and continuing.
    checkpoint_interval: int = 1000
    # Replay buffers must be checkpointed together with the model so that
    # global_step and buffer/online_size stay consistent after resuming.
    save_replay_on_checkpoint: bool = True
    # Normally require the replay sidecar paired with resume_checkpoint.
    # Set False only for recovery when that sidecar was lost; the most recent
    # compatible generic online cache will be used instead, while future
    # checkpoints can still save exact replay sidecars.
    require_exact_replay_on_resume: bool = True
    # Explicitly reuse an existing run_<timestamp>_<name> directory even when
    # restarting before the first checkpoint has been created.
    run_output_dir: str | None = None
    # Keep checkpoints and artifacts in the original run directory when
    # resuming, instead of creating run_<new timestamp>_<name> each time.
    reuse_run_dir_on_resume: bool = True
    # Checkpoints are required for future resume, so preserve the run directory
    # after a normal training completion unless cleanup is explicitly requested.
    cleanup_run_dir_on_finish: bool = False
    save_online_rb_interval: int = 1000
    send_transitions_len : int = 1
    chunk_len: int = 1

@dataclass
class ResidualTD3FrankaComplexConfig(ResidualTD3DexmgConfig):

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    # Ablation: remove only low-success task prioritization. The residual gate
    # remains controlled independently by agent.gate_mode/use_residual_gate.
    prioritize_low_success_tasks: bool = False

    # Reuse this completed 8-task evaluation for every fresh complex-task run.
    initial_task_success_log: str | None = (
        "/home/yuan/self_vla/residual-offpolicy-rl/outputs/"
        "task_reward_generation/20260901_185152/step_0/log.txt"
    )
    initial_task_success_window_overrides: dict[str, list[int]] | None = field(


        default_factory=lambda: {
            "Open the drawer": [
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 1,
            ],
            "Put a cube into the bowl": [
                1, 1, 0, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
            "Stack one cube on the other cube": [
                1, 1, 0, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
            "Take the mug off the mug tree": [
                1, 1, 0, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
        }
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="franka-complex-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="/home/yuan/self_vla/tele_op/lerobot/dataset_10070_10070_5050_6050",
            num_episodes=550,
            horizon=400,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )

@dataclass
class ResidualTD3FrankaCubeConfig(ResidualTD3DexmgConfig):
    task: str = "pick up the cube and place it into the bowl"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="franka-cube-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="/home/yuan/self_vla/tele_op/lerobot/cube_fix_in50_out30",
            num_episodes=1_000,
            horizon=140,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


# Register with Hydra
cs = ConfigStore.instance()
cs.store(name="residual_td3_dexmg_config", node=ResidualTD3DexmgConfig)
cs.store(name="residual_td3_franka_complex_config", node=ResidualTD3FrankaComplexConfig)
cs.store(name="residual_td3_franka_cube_config", node=ResidualTD3FrankaCubeConfig)
