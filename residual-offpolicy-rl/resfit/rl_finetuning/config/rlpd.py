# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from torch import nn


@dataclass
class VitEncoderConfig:
    depth: int = 1
    embed_dim: int = 128
    embed_norm: int = 0
    embed_style: str = "embed2"
    num_heads: int = 4
    patch_size: int = 8
    stride: int = -1
    act_layer = nn.GELU

@dataclass
class SiglipEncoderConfig:
    model_name: str = "google/siglip-base-patch16-224"
    freeze: bool = True
    force_image_size: int = 224
    use_processor_norm: bool = False
    drop_cls_token: bool = False


@dataclass
class CriticLossCfg:
    type: str = "mse"
    n_bins: int = 51  # 0.995 γ → δ=0.005  # noqa: RUF003
    v_min: float = 0.0
    v_max: float = 1.0
    sigma: float = -1.0  # Use -1 to indicate auto-compute: 0.75 * (v_max - v_min) / n_bins

    def __post_init__(self):
        assert self.type in ["mse", "hl_gauss", "c51"]


@dataclass
class CriticConfig:
    drop: float = 0.2
    feature_dim: int = 128
    fuse_patch: int = 1
    hidden_dim: int = 1024
    norm_weight: int = 0
    orth: int = 1
    spatial_emb: int = 1024

    # Number of independent Q-heads (in the ensemble). Set to 2 for TD3, >2 for RED-Q style.
    num_q: int = 10
    loss: CriticLossCfg = field(default_factory=lambda: CriticLossCfg())
    # Policy gradient type: "ensemble_mean" (mean over all heads, standard RED-Q) or
    policy_gradient_type: str = "ensemble_mean"
    # Number of hidden layers in the critic MLP heads (default 2 for backwards compatibility)
    num_layers: int =4
    # Layer normalization control
    use_layer_norm: bool = True
    # Number of Q-heads to take min over for target computation (default 2 for RED-Q behavior)
    min_q_heads: int = 2

    def __post_init__(self):
        assert self.policy_gradient_type in ["ensemble_mean", "min_random_pair", "q1"], (
            f"Invalid policy_gradient_type: {self.policy_gradient_type}"
        )

@dataclass
class LanguageConfig:
    enabled: bool = True

    lang_emb_obs_key: str = "observation.task_emb"

    lang_emb_dim: int = 384
    lang_proj_dim: int = 64

    hidden_dim: int = 256
    num_layers: int = 2
    use_layer_norm: bool = True
    dropout: float = 0.0

    fusion: str = "prop"

    def __post_init__(self):
        assert self.fusion in ("prop",), (
            f"Only 'prop' fusion is implemented in this variant, got {self.fusion!r}"
        )

@dataclass
class ActorConfig:
    feature_dim: int = 128
    hidden_dim: int = 1024
    dropout: float = 0.2
    orth: int = 1
    max_action_norm: float = -1
    spatial_emb: int = 0

    # Optional parameter to control last layer initialization scale
    # If None, uses default initialization. If set to a small value (e.g., 1e-3):
    # - For 'normal': used as standard deviation
    # - For 'orthogonal'/'xavier_uniform': used as gain to scale initialization close to zero
    actor_last_layer_init_scale: Optional[float] = None
    # Distribution to use for last layer initialization
    # Options: 'normal', 'orthogonal', 'xavier_uniform'
    actor_last_layer_init_distribution: str = "normal"
    # Distribution to use for intermediate layer initialization
    # Options: 'default', 'normal', 'orthogonal', 'xavier_uniform'
    actor_intermediate_layer_init_distribution: str = "default"
    # L2 regularization weight on action magnitude
    action_l2_reg_weight: float = 0.0
    action_scale: float = 1.0
    # Number of hidden layers in the actor MLP (default 2 for backwards compatibility)
    num_layers: int = 4
    # Layer normalization control
    use_layer_norm: bool = True

    scale_head_max: float = 1.0 * 0.1
    scale_head_init_value: float = 0.01
    scale_head_per_dim: bool = True
    scale_head_enabled: bool = False

@dataclass
class QAgentConfig:
    device: str = "cuda"
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_clip: float = 0.3
    # LR warmup configuration
    lr_warmup_steps: int = 0  # Number of warmup steps
    lr_warmup_start: float = 1e-8  # Starting LR for warmup (very small positive value)
    # encoder
    use_prop: int = 1
    enc_type: str = "vit"
    vit: VitEncoderConfig = field(default_factory=lambda: VitEncoderConfig())
    siglip: SiglipEncoderConfig = field(default_factory=SiglipEncoderConfig)
    # critic & actor
    critic: CriticConfig = field(default_factory=lambda: CriticConfig())
    actor: ActorConfig = field(default_factory=lambda: ActorConfig())
    language: LanguageConfig | None = field(default_factory=LanguageConfig)
    # Optional hard actor routing. The image/RL-token encoder and critic stay
    # shared, while every discrete task owns an independent residual actor and
    # actor target.
    task_specific_actor: bool = False
    num_tasks: int = 1
    task_id_obs_key: str = "observation.task_id"
    # gradient clipping
    critic_grad_clip_norm: float = 1.0
    actor_grad_clip_norm: float = 0.1

    # Temporal regularization for flattened action chunks. The trainer sets
    # action_chunk_len from its top-level chunk_len before constructing agents.
    action_chunk_len: int = 1
    executed_residual_dims: int = 3
    temporal_smoothness_dims: int = 3
    temporal_smoothness_intra_weight: float = 0
    temporal_smoothness_boundary_weight: float = 0

    # bc loss regularization
    bc_loss_coef: float = 0.0
    bc_loss_dynamic: int = 0  # dynamically scale bc loss weight
    bc_backprop_encoder: bool = False  # Whether BC loss should update encoder (default False for RLPD)

    # encoder freezing
    freeze_encoder: bool = False  # Whether to freeze encoder parameters (no gradient updates)

    clip_q_target_to_reward_range: bool = True

    # TD3 target action noise configuration
    target_action_noise: bool = True  # Whether to add noise to target actions in TD3

    # When enabled, the residual (combined action) is only adopted where the critic
    # judges it a real improvement over the base action, i.e.
    # Q(s, a_base + residual) - Q(s, a_base) > gate_threshold; otherwise the base
    # action is executed (residual -> 0). Applied both when acting and in the critic
    # target's next-action selection. Default off (existing behavior).
    gate_mode: str = "hard"  # "off" or "hard"
    use_residual_gate: bool = False  # alternative switch; "hard" gate_mode also enables it
    gate_threshold: float = 0.0  # Q-advantage threshold for adopting the residual

    def __post_init__(self):
        if self.num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {self.num_tasks}")


# Algorithm-specific hyper-parameters -----------------------------------------


@dataclass
class RLPDAlgoConfig:
    """RLPD hyper-parameters."""

    # Training ---------------------------------------------------------------
    total_timesteps: int = 300_000
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 10_000

    # Discounting / target updates ------------------------------------------
    gamma: float = 0.99

    # Update scheduling ------------------------------------------------------
    num_updates_per_iteration: int = 16
    actor_updates_per_iteration: int = 4
    update_every_n_steps: int = 1

    # Offline / online mixture ----------------------------------------------
    offline_fraction: float = 0.5  # Fraction of minibatch sampled from the offline buffer.

    # N-step returns ----------------------------------------------------
    # Horizon used by the MultiStep transform in our replay buffers.
    n_step: int = 3

    # Number of critic-only updates before training the actor
    critic_warmup_steps: int = 0

    # Actor learning rate warmup schedule -----------------------------
    # Number of steps over which to linearly warm up the actor learning rate from 0 to full actor_lr
    actor_lr_warmup_steps: int = 0

    # Optional Behavioural Cloning (BC) loss ---------------------------

    # Scale for random action noise during initial exploration phase
    # Actions are sampled as: rand_actions = torch.rand(...) * 2 * random_action_noise_scale - random_action_noise_scale
    random_action_noise_scale: float = 1.0  # Default: uniform in [-1, 1]

    # Sampling strategy -------------------------------------------------
    # Sampling strategy for replay buffer:
    sampling_strategy: str = "uniform"

    # Prefetch batches using background threads to speed up sampling
    prefetch_batches: int = 4  # Number of batches to prefetch (0 = disabled, 4-8 recommended)

    # Priority parameters for prioritized experience replay
    priority_alpha: float = 0.6  # Controls prioritization strength (0=uniform, higher=more prioritization)
    priority_beta: float = 0.4  # Controls importance sampling correction

    stddev_max: float = 0.1
    stddev_min: float = 0.1
    stddev_step: int = 300_000

    stddev_schedule: str = field(init=False)

    def __post_init__(self):
        self.stddev_schedule = f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


# Dataset specification --------------------------------------------------------


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: Optional[int] = 300
    image_key: Optional[str] = None


@dataclass
class WandBConfig:
    project: str = "rlpd"
    name: Optional[str]= None
    mode = "online"
    entity:Optional[str] = None
    notes:Optional[str] = None
    continue_run_id:Optional[str] = None
    group:Optional[str] = None

@dataclass
class RLPDDexmgConfig:
    # General

    seed: Optional[int] = None
    torch_deterministic: bool = False
    debug: bool = False


    # Task / environment
    task: str = "Can"
    num_envs: int = 1
    eval_num_envs: int = 8
    eval_num_episodes: int = 20
    headless: bool = True
    video_key: str = "observation.images.agentview"
    rl_camera: List[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ]
    )

    algo: RLPDAlgoConfig = field(default_factory=RLPDAlgoConfig)

    agent: QAgentConfig = field(default_factory=QAgentConfig)

    offline_data: OfflineDataConfig = field(default_factory=OfflineDataConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    log_freq: int = 100
    eval_interval_every_steps: int = 10_000
    checkpoint_interval: int = -1
    save_video: bool = True
    # Whether to run an evaluation pass before training begins (at step 0)
    eval_first: bool = True
