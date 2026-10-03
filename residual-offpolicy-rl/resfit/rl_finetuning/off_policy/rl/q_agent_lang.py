# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy import common_utils
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.networks.encoder import VitEncoder
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.off_policy.rl.critic import Critic
from resfit.rl_finetuning.off_policy.rl.lang_encoder import LanguageEncoder


def _validated_task_ids(
    task_ids: torch.Tensor,
    *,
    batch_size: int,
    num_tasks: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a scalar/[B]/[B,1] task id tensor and validate its range."""
    ids = torch.as_tensor(task_ids, device=device)
    if ids.is_floating_point():
        rounded = ids.round()
        if not torch.equal(ids, rounded):
            raise ValueError(f"task ids must be integers, got {ids.detach().cpu().tolist()}")
        ids = rounded
    ids = ids.to(dtype=torch.long).reshape(-1)
    if ids.numel() == 1 and batch_size > 1:
        ids = ids.expand(batch_size)
    if ids.numel() != batch_size:
        raise ValueError(
            f"Expected {batch_size} task ids, got shape {tuple(task_ids.shape)}"
        )
    if ids.numel() and (ids.min().item() < 0 or ids.max().item() >= num_tasks):
        raise ValueError(
            f"task ids must be in [0, {num_tasks - 1}], got "
            f"{sorted(ids.unique().detach().cpu().tolist())}"
        )
    return ids


def _select_obs_rows(
    obs: dict[str, torch.Tensor], indices: torch.Tensor, batch_size: int
) -> dict[str, torch.Tensor]:
    selected = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            selected[key] = value.index_select(0, indices)
        else:
            selected[key] = value
    return selected


class TaskSpecificActor(nn.Module):
    """Batch router over fully independent residual actors."""

    def __init__(self, actors: list[Actor], task_id_obs_key: str):
        super().__init__()
        if not actors:
            raise ValueError("TaskSpecificActor requires at least one actor")
        self.actors = nn.ModuleList(actors)
        self.task_id_obs_key = task_id_obs_key
        self.last_sigma: torch.Tensor | None = None

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        feat = obs["feat"]
        batch_size = feat.shape[0]
        if self.task_id_obs_key not in obs:
            raise KeyError(
                f"Task-specific actor requires '{self.task_id_obs_key}' in every observation"
            )
        task_ids = _validated_task_ids(
            obs[self.task_id_obs_key],
            batch_size=batch_size,
            num_tasks=len(self.actors),
            device=feat.device,
        )

        loc = None
        scale = None
        routed_sigma = None
        for task_id in task_ids.unique(sorted=True).tolist():
            indices = (task_ids == task_id).nonzero(as_tuple=False).flatten()
            task_obs = _select_obs_rows(obs, indices, batch_size)
            task_dist = self.actors[task_id](task_obs, std)
            if loc is None:
                loc = task_dist.loc.new_zeros((batch_size, task_dist.loc.shape[-1]))
                scale = task_dist.scale.new_zeros((batch_size, task_dist.scale.shape[-1]))
            loc = loc.index_copy(0, indices, task_dist.loc)
            scale = scale.index_copy(0, indices, task_dist.scale)

            sigma = self.actors[task_id].last_sigma
            if sigma is not None:
                if routed_sigma is None:
                    routed_sigma = sigma.new_zeros((batch_size, sigma.shape[-1]))
                routed_sigma = routed_sigma.index_copy(0, indices, sigma)

        if loc is None or scale is None:
            raise RuntimeError("Cannot route an empty actor batch")
        self.last_sigma = routed_sigma
        return utils.TruncatedNormal(loc, scale)


class QAgentLang(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
        action_dim: int,
        rl_cameras: list[str] | str,
        cfg: QAgentConfig,
        residual_actor: bool = False,
    ):
        """Initialize the language-conditioned Q-agent.

        The proprioceptive vector seen by the actor / critic is
        ``concat(observation.state, lang_proj(observation.task_emb))`` so the
        underlying networks are unchanged structurally — they just see a
        slightly larger ``prop_dim``.

        Parameters
        ----------
        obs_shape : tuple[int, int, int]
            Shape (C, H, W) for **a single camera** image.
        prop_shape : tuple[int]
            Shape of the *raw* proprioceptive vector (before task embedding
            is appended).
        action_dim : int
            Number of action dimensions.
        rl_cameras : list[str] | str
            Name(s) of the camera images to be used by the RL policy.
        cfg : QAgentConfig
            Hyper-parameter configuration dataclass; must include
            ``cfg.language``.
        residual_actor : bool
            Whether the actor outputs a residual on top of a base action.
        """
        super().__init__()
        if isinstance(rl_cameras, str):
            rl_cameras = [rl_cameras]
        assert len(rl_cameras) > 0, "At least one camera must be provided"

        self.rl_cameras = rl_cameras
        self.cfg = cfg
        self.residual_actor = residual_actor
        self.task_specific_actor = bool(getattr(cfg, "task_specific_actor", False))
        self.num_tasks = int(getattr(cfg, "num_tasks", 1))
        self.task_id_obs_key = str(
            getattr(cfg, "task_id_obs_key", "observation.task_id")
        )
        if self.task_specific_actor and self.num_tasks < 2:
            raise ValueError(
                "task_specific_actor requires num_tasks >= 2, "
                f"got {self.num_tasks}"
            )
        self.action_chunk_len = int(getattr(cfg, "action_chunk_len", 1))
        if self.action_chunk_len < 1 or action_dim % self.action_chunk_len != 0:
            raise ValueError(
                f"action_dim={action_dim} must be divisible by "
                f"action_chunk_len={self.action_chunk_len}"
            )
        self.per_step_action_dim = action_dim // self.action_chunk_len
        self.executed_residual_dims = max(
            0,
            min(
                int(getattr(cfg, "executed_residual_dims", 3)),
                self.per_step_action_dim,
            ),
        )
        self.temporal_smoothness_dims = max(
            0,
            min(
                int(getattr(cfg, "temporal_smoothness_dims", 3)),
                self.per_step_action_dim,
            ),
        )

        # Adopt the combined (base + residual) action only where the critic judges it a
        # clear improvement over the base action; otherwise fall back to the base action
        # (residual -> 0). Applied in act() and in the critic target's next-action.
        _gate_mode = str(getattr(cfg, "gate_mode", "off"))
        self.use_residual_gate = bool(getattr(cfg, "use_residual_gate", False)) or (_gate_mode == "hard")
        self.residual_gate_threshold = float(getattr(cfg, "gate_threshold", 0.0))
        # Most-recent gate adoption rate (logged), split by phase.
        self.last_rollout_gate_adopt_rate = None
        self.last_eval_gate_adopt_rate = None
        self._last_gate_adv = None

        self.lang_cfg = cfg.language
        self.lang_enabled: bool = bool(self.lang_cfg.enabled)
        self._raw_prop_dim: int = int(prop_shape[0])

        self.encoders: nn.ModuleList = self._build_encoders(obs_shape)

        sample_encoder = self.encoders[0]
        repr_dim_single = int(sample_encoder.repr_dim)  # type: ignore[attr-defined]
        patch_repr_dim = int(sample_encoder.patch_repr_dim)  # type: ignore[attr-defined]

        repr_dim = repr_dim_single * len(self.rl_cameras)
        print("encoder output dim: ", repr_dim)
        print("patch output dim: ", patch_repr_dim)

        assert len(prop_shape) == 1
        raw_prop_dim = prop_shape[0] if cfg.use_prop else 0

        if self.lang_enabled:
            self.lang_encoder = LanguageEncoder(
                lang_emb_dim=self.lang_cfg.lang_emb_dim,
                lang_proj_dim=self.lang_cfg.lang_proj_dim,
                hidden_dim=self.lang_cfg.hidden_dim,
                num_layers=self.lang_cfg.num_layers,
                use_layer_norm=self.lang_cfg.use_layer_norm,
                dropout=self.lang_cfg.dropout,
            )
            prop_dim = raw_prop_dim + self.lang_cfg.lang_proj_dim
        else:
            self.lang_encoder = None
            prop_dim = raw_prop_dim
        self._effective_prop_dim: int = int(prop_dim)

        def make_critic():
            return Critic(
                repr_dim=repr_dim,
                patch_repr_dim=patch_repr_dim,
                prop_dim=prop_dim,
                action_dim=action_dim,
                cfg=self.cfg.critic,
            )

        def make_actor():
            return Actor(
                repr_dim,
                patch_repr_dim,
                prop_dim,
                action_dim,
                cfg.actor,
                residual_actor=residual_actor,
            )

        self.critic = make_critic()
        if self.task_specific_actor:
            self.actor = TaskSpecificActor(
                [make_actor() for _ in range(self.num_tasks)],
                self.task_id_obs_key,
            )
        else:
            self.actor = make_actor()

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        print(common_utils.wrap_ruler("encoder weights"))
        print(self.encoders)
        common_utils.count_parameters(self.encoders)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        # Freeze encoder parameters if requested
        if getattr(self.cfg, "freeze_encoder", False):
            for param in self.encoders.parameters():
                param.requires_grad = False
            print("🧊 Encoder parameters frozen - no gradient updates will be performed")

        # Create optimizers (PyTorch will ignore frozen parameters).
        # The language encoder is trained jointly with the critic, so its
        # parameters live in ``critic_opt``; during the actor update we detach
        # the projected task feature so its gradient does not flow into the
        # language encoder there.
        self.encoder_opt = torch.optim.AdamW(self.encoders.parameters(), lr=self.cfg.critic_lr)
        critic_params = list(self.critic.parameters())
        if self.lang_enabled:
            critic_params += list(self.lang_encoder.parameters())
        self.critic_opt = torch.optim.AdamW(critic_params, lr=self.cfg.critic_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor_lr)

        # LR schedulers for warmup (if warmup is enabled)
        self.encoder_scheduler = None
        self.critic_scheduler = None
        self.actor_scheduler = None

        if self.cfg.lr_warmup_steps > 0:
            # LinearLR scheduler that linearly ramps from start_factor to 1.0 over total_iters steps
            # Note: start_factor must be > 0 for LinearLR scheduler
            warmup_start = self.cfg.lr_warmup_start

            # Calculate start factors for each optimizer
            critic_start_factor = warmup_start / self.cfg.critic_lr if self.cfg.critic_lr > 0 else 1e-8
            critic_start_factor = max(critic_start_factor, 1e-8)

            actor_start_factor = warmup_start / self.cfg.actor_lr if self.cfg.actor_lr > 0 else 1e-8
            actor_start_factor = max(actor_start_factor, 1e-8)

            # Create schedulers with appropriate start factors
            self.encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.encoder_opt, start_factor=critic_start_factor, total_iters=self.cfg.lr_warmup_steps
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_opt, start_factor=critic_start_factor, total_iters=self.cfg.lr_warmup_steps
            )
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_opt, start_factor=actor_start_factor, total_iters=self.cfg.lr_warmup_steps
            )

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        self.bc_policies: list[nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: common_utils.MultiCounter | None = None

        self.critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)

    def _augment_state(
        self, obs: dict[str, torch.Tensor], *, detach_lang: bool = False
    ) -> dict[str, torch.Tensor]:
        """Return a *shallow copy* of ``obs`` whose ``observation.state`` has
        been concatenated with the projected task embedding.

        Idempotent: if ``observation.state`` already has the augmented size,
        the obs is returned unchanged.
        """
        if not self.lang_enabled:
            return obs

        state = obs["observation.state"]
        # Idempotent guard so the caller can safely double-augment.
        if state.size(-1) == self._effective_prop_dim:
            return obs
        assert state.size(-1) == self._raw_prop_dim, (
            f"Unexpected observation.state size {state.size(-1)}, "
            f"expected raw {self._raw_prop_dim} or augmented {self._effective_prop_dim}"
        )

        key = self.lang_cfg.lang_emb_obs_key
        assert key in obs, (
            f"Language fusion enabled but obs is missing '{key}'. "
            f"Make sure the env wrapper / replay buffer populates it."
        )

        x = obs[key]
        feat = self.lang_encoder(x)  # [B, lang_proj_dim]
        if detach_lang:
            feat = feat.detach()

        # Broadcast batch dims
        if state.dim() == 2 and feat.dim() == 2:
            if feat.size(0) == 1 and state.size(0) > 1:
                feat = feat.expand(state.size(0), -1)
        elif state.dim() == 1 and feat.dim() == 2 and feat.size(0) == 1:
            feat = feat.squeeze(0)
        elif state.dim() == 2 and feat.dim() == 1:
            feat = feat.unsqueeze(0).expand(state.size(0), -1)

        new_state = torch.cat([state, feat], dim=-1)

        new_obs = copy.copy(obs)
        new_obs["observation.state"] = new_state
        return new_obs

    def _critic_forward(self, critic, obs, action, *, return_logits=False):
        return critic(
            obs["feat"],
            obs["observation.state"],
            action,
            return_logits=return_logits,
        )

    def _critic_q_value(self, critic, obs, action):
        return critic.q_value(obs["feat"], obs["observation.state"], action)

    def _critic_q_value_for_policy(self, obs, action):
        return self.critic.q_value_for_policy(
            obs["feat"], obs["observation.state"], action
        )

    @torch.no_grad()
    def predict_q(self, obs: dict[str, torch.Tensor], action: torch.Tensor):
        """Public shared-critic Q query used by evaluation/diagnostics."""
        return self._critic_q_value(self.critic, obs, action)

    def _soft_update_task_networks(self, source, target, obs):
        """Update the shared critic or only the actor heads in this batch."""
        if not isinstance(source, TaskSpecificActor):
            utils.soft_update_params(source, target, self.cfg.critic_target_tau)
            return
        task_ids = _validated_task_ids(
            obs[self.task_id_obs_key],
            batch_size=obs["feat"].shape[0],
            num_tasks=self.num_tasks,
            device=obs["feat"].device,
        )
        for task_id in task_ids.unique().tolist():
            utils.soft_update_params(
                source.actors[task_id],
                target.actors[task_id],
                self.cfg.critic_target_tau,
            )

    def _build_encoders(self, obs_shape):
        """Constructs and returns an ``nn.ModuleList`` with one encoder per
        camera based on ``self.cfg.enc_type``.  All encoders share the same
        architecture and therefore yield feature tensors with identical
        dimensions which simplifies feature fusion downstream.
        """

        encoders = nn.ModuleList()

        for _ in self.rl_cameras:
            if self.cfg.enc_type == "vit":
                enc = VitEncoder(obs_shape, self.cfg.vit).to(self.cfg.device)
            else:
                raise AssertionError(f"Unknown encoder type {self.cfg.enc_type}.")

            encoders.append(enc)

        return encoders

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        self.encoders.train(training)
        self.actor.train(training)
        self.critic.train(training)
        if self.lang_encoder is not None:
            self.lang_encoder.train(training)

        assert not self.critic_target.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    def _min_over_random_two(self, q_values: torch.Tensor) -> torch.Tensor:
        """Compute min over a random subset of 2 heads from q_values [K, B, 1]."""
        assert q_values.dim() == 3 and q_values.size(-1) == 1
        num_heads = q_values.size(0)
        if num_heads <= 2:
            return torch.min(q_values[0], q_values[1])
        # Sample two unique head indices uniformly at random
        idx = torch.randperm(num_heads, device=q_values.device)[:2]
        subset = q_values.index_select(dim=0, index=idx)  # [2, B, 1]
        return subset.min(dim=0).values

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.cfg.act_method
        assert original_method != override_method

        self.cfg.act_method = override_method
        yield

        self.cfg.act_method = original_method

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        r"""This function encodes the observation into feature tensor.

        Images may be stored in the replay buffers as uint8 to save GPU memory.  In
        that case we convert them to float32 in \[0,1] before feeding them to the
        encoders.  If the image is already a float tensor (offline dataset or
        direct env observations during evaluation) we assume it is properly
        normalised.
        """
        feats = []
        for cam_idx, cam_name in enumerate(self.rl_cameras):
            data = obs[cam_name]

            if data.dtype == torch.uint8:
                # uint8 → float32 in [0,1]
                data = data.float().div_(255.0)
            else:
                data = data.float()

            if augment:
                data = self.aug(data)

            # Forward pass through the *corresponding* encoder
            feat_cam = self.encoders[cam_idx].forward(data, flatten=False)
            feats.append(feat_cam)

        # Concatenate along the *patch* dimension (dim=1)
        feat_all = torch.cat(feats, dim=1)
        return feat_all  # noqa: RET504

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if obs[self.rl_cameras[0]].dim() == 3:
            should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    @torch.no_grad()
    def _gate_adopt_mask(self, obs, base_action, combined, *, use_target: bool = False):
        """Return ``(adopt, adv)`` where ``adopt`` is a (B, 1) bool mask that is True
        where the combined action is adopted, i.e. the critic's advantage
        ``Q(comb) - Q(base) > residual_gate_threshold``.

        Uses the ensemble-mean Q (same deterministic estimator for base and comb) so
        the comparison is consistent.
        """
        critic = self.critic_target if use_target else self.critic
        q_base = self._critic_forward(critic, obs, base_action).mean(dim=0).reshape(-1)
        q_comb = self._critic_forward(critic, obs, combined).mean(dim=0).reshape(-1)
        adv = q_comb - q_base                                            # (B,)
        adopt = adv > self.residual_gate_threshold                       # (B,)
        if not use_target:  # acting path -> stash for eval logging
            self._last_gate_adv = adv.detach()
        return adopt.view(-1, 1), adv

    def act(self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        # Make a shallow copy of the observation dict
        obs = copy.copy(obs)
        unsqueezed = self._maybe_unsqueeze_(obs)

        # Inject the projected task feature into observation.state.
        # detach_lang=True since we never train the lang encoder via act().
        obs = self._augment_state(obs, detach_lang=True)

        assert "feat" not in obs
        obs["feat"] = self._encode(obs, augment=False)

        action = self._act_default(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            clip=1,  # clip
            use_target=False,
        )
        if self.residual_actor:
            action = self._mask_unexecuted_residual_dims(action)

        # Hard Q-advantage gate: keep the base action (residual -> 0) unless the
        # combined action is a clear improvement over the base action.
        if self.use_residual_gate and self.residual_actor:
            base_action = obs["observation.base_action"]
            combined = torch.clamp(base_action + action, -1.0, 1.0)
            adopt, _adv = self._gate_adopt_mask(
                obs, base_action, combined, use_target=False
            )
            adopt_rate = adopt.float().mean().item()
            if eval_mode:
                self.last_eval_gate_adopt_rate = adopt_rate
            else:
                self.last_rollout_gate_adopt_rate = adopt_rate
            action = action * adopt.to(action.dtype)

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def _mask_unexecuted_residual_dims(self, residual: torch.Tensor) -> torch.Tensor:
        """Match training and inference to the XYZ-only residual executed by BasePolicy."""
        chunk = residual.reshape(
            residual.shape[0], self.action_chunk_len, self.per_step_action_dim
        )
        mask = torch.zeros_like(chunk)
        mask[..., : self.executed_residual_dims] = 1.0
        return (chunk * mask).reshape_as(residual)

    def _act_default(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: float | None,
        use_target: bool,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        dist = actor.forward(obs, stddev)

        # Only assert not training when this is called from the public act() method
        # (which is used for actual evaluation), not when called internally during training
        if eval_mode and not use_target:
            assert not self.training

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=clip)

        return action

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
        importance_weights: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            # use train mode as we use actor dropout
            assert self.actor_target.training

            # Predict next residual action and form the combined next action
            # Use target_action_noise config to control whether to add noise to target actions
            next_residual_action = self._act_default(
                obs=next_obs,
                eval_mode=not self.cfg.target_action_noise,  # Disable noise if target_action_noise=False
                stddev=stddev,
                clip=self.cfg.stddev_clip,
                use_target=True,
            )
            if self.residual_actor:
                next_residual_action = self._mask_unexecuted_residual_dims(
                    next_residual_action
                )

            if self.residual_actor:
                # Current step: 'action' from the replay buffer is the executed combined action
                # Next step: combine and clamp to match environment execution
                next_base_action = next_obs["observation.base_action"]
                next_action = torch.clamp(next_base_action + next_residual_action, -1.0, 1.0)
                # Gate the bootstrap action consistently with acting: fall back to the
                # base action where the combined one is not a clear improvement.
                if self.use_residual_gate:
                    adopt, _adv = self._gate_adopt_mask(
                        next_obs, next_base_action, next_action, use_target=True,
                    )
                    next_action = torch.where(adopt, next_action, next_base_action)
                    self._gate_adopt_rate = adopt.float().mean().item()
            else:
                next_action = next_residual_action

            # Compute target Q using min over a random subset of 2 heads
            target_all = self._critic_q_value(
                self.critic_target, next_obs, next_action
            )
            target_q_min = target_all.squeeze(-1)  # [B]
            target_q = (reward + (discount * target_q_min)).detach()

        if self.cfg.clip_q_target_to_reward_range:
            target_q = torch.clamp(target_q, min=0, max=1)  # Sparse rewards are in {0, 1}

        td_errors = None

        if self.critic.loss_cfg.type == "hl_gauss":
            # Compute logits for current Q heads and average HL-Gauss loss across heads
            q_per_head, logits_per_head = self._critic_forward(
                self.critic, obs, action, return_logits=True
            )
            K = logits_per_head.shape[0]
            losses = [self.critic.hl_loss(logits_per_head[i], target_q) for i in range(K)]
            critic_loss = torch.stack(losses).mean()
        elif self.critic.loss_cfg.type == "c51":
            # Compute logits for current Q heads and C51 distributional loss
            q_per_head, logits_per_head = self._critic_forward(
                self.critic, obs, action, return_logits=True
            )

            # Get next state distribution for C51 target computation
            with torch.no_grad():
                _, next_logits = self._critic_forward(
                    self.critic_target, next_obs, next_action, return_logits=True
                )
                # Take min over random subset of heads for next distribution (configurable via min_q_heads)
                num_heads = min(self.critic.cfg.min_q_heads, next_logits.shape[0])
                idx = torch.randperm(next_logits.shape[0], device=next_logits.device)[:num_heads]
                next_logits_min = torch.min(next_logits.index_select(0, idx), dim=0).values
                next_distribution = torch.softmax(next_logits_min, dim=-1)

                # Project the target distribution
                # The discount factor passed to this function is already discount = gamma * (1 - done)
                # For C51, we need to extract the done mask and gamma separately
                # We'll use a simple heuristic: if discount is 0, then done=1, otherwise done=0
                dones = (discount == 0.0).float()
                gamma = 0.99  # Assume standard gamma value
                target_distribution = self.critic.c51_loss.project_distribution(next_distribution, reward, dones, gamma)

            # Compute C51 loss for each head
            K = logits_per_head.shape[0]
            losses = [self.critic.c51_loss(logits_per_head[i], target_distribution) for i in range(K)]
            critic_loss = torch.stack(losses).mean()
        else:
            q_all = self._critic_forward(self.critic, obs, action).squeeze(-1)  # [K,B]
            # Compute TD errors for prioritized experience replay (before taking mean)
            td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B] - mean across heads

            # Apply importance sampling weights if provided (for prioritized experience replay)
            if importance_weights is not None:
                # Weight the squared TD errors by importance sampling weights
                weighted_td_errors = td_errors**2 * importance_weights
                critic_loss = weighted_td_errors.mean()
            else:
                # Mean squared error across heads and batch (uniform sampling)
                critic_loss = (td_errors**2).mean()

        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()
        if self.use_residual_gate and getattr(self, "_gate_adopt_rate", None) is not None:
            metrics["train/gate_adopt_rate"] = self._gate_adopt_rate
        # Store target_q for potential logging (calculated only when needed)
        metrics["_target_q"] = target_q.detach().cpu()
        # Store TD errors for prioritized experience replay
        if td_errors is not None:
            metrics["_td_errors"] = td_errors.detach().cpu()
        # Log importance sampling weights for monitoring PER behavior
        if importance_weights is not None:
            metrics["train/importance_weights_mean"] = importance_weights.mean().item()
            metrics["train/importance_weights_std"] = importance_weights.std().item()
            metrics["train/importance_weights_min"] = importance_weights.min().item()
            metrics["train/importance_weights_max"] = importance_weights.max().item()

        # Zero gradients
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        # Gradient clipping
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self.encoders.parameters(), self.cfg.critic_grad_clip_norm)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)

        # Store gradient norms for logging
        metrics["train/encoder_grad_norm"] = encoder_grad_norm.item()
        metrics["train/critic_grad_norm"] = critic_grad_norm.item()

        self.encoder_opt.step()
        self.critic_opt.step()

        return metrics

    def _temporal_smoothness_losses(
        self,
        action_pred: torch.Tensor,
        prev_residual_last: torch.Tensor | None,
        has_prev_residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unweighted intra-chunk and cross-chunk residual MSE losses."""
        residual_chunk = action_pred.reshape(
            action_pred.shape[0], self.action_chunk_len, self.per_step_action_dim
        )
        smooth_dims = self.temporal_smoothness_dims

        if self.action_chunk_len > 1 and smooth_dims > 0:
            intra_delta = (
                residual_chunk[:, 1:, :smooth_dims]
                - residual_chunk[:, :-1, :smooth_dims]
            )
            intra_smoothness_loss = intra_delta.square().mean()
        else:
            intra_smoothness_loss = action_pred.new_zeros(())

        boundary_smoothness_loss = action_pred.new_zeros(())
        if prev_residual_last is not None and has_prev_residual is not None and smooth_dims > 0:
            previous = prev_residual_last.to(action_pred.device).reshape(
                action_pred.shape[0], -1
            )[:, :smooth_dims]
            valid = has_prev_residual.to(action_pred.device).reshape(-1).bool()
            if valid.any():
                boundary_delta = residual_chunk[:, 0, :smooth_dims] - previous
                boundary_smoothness_loss = boundary_delta[valid].square().mean()
        return intra_smoothness_loss, boundary_smoothness_loss

    def _compute_actor_loss(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        prev_residual_last: torch.Tensor | None = None,
        has_prev_residual: torch.Tensor | None = None,
    ):
        assert "feat" in obs, "safety check"

        action_pred: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            # NOTE: This fix has not been fully verified yet.
            stddev=0.0,
            clip=self.cfg.stddev_clip,
            use_target=False,
        )
        if self.residual_actor:
            action_pred = self._mask_unexecuted_residual_dims(action_pred)

        # Add L2 regularization on action magnitude (before we add the residual action to the base)
        action_l2_penalty = self.cfg.actor.action_l2_reg_weight * torch.mean(torch.sum(action_pred**2, dim=-1))

        if self.residual_actor:
            # Create the full action by combining the base action and the residual action
            # and clamp to the valid range [-1, 1] to match environment execution
            combined_action = torch.clamp(obs["observation.base_action"] + action_pred, -1.0, 1.0)
        else:
            combined_action = action_pred

        q = self._critic_q_value_for_policy(obs, combined_action)
        actor_loss_base = -q.mean()

        intra_smoothness_loss, boundary_smoothness_loss = (
            self._temporal_smoothness_losses(
                action_pred, prev_residual_last, has_prev_residual
            )
        )

        intra_penalty = (
            float(getattr(self.cfg, "temporal_smoothness_intra_weight", 0.0))
            * intra_smoothness_loss
        )
        boundary_penalty = (
            float(getattr(self.cfg, "temporal_smoothness_boundary_weight", 0.0))
            * boundary_smoothness_loss
        )
        actor_loss_total = (
            actor_loss_base + action_l2_penalty + intra_penalty + boundary_penalty
        )

        return (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
            intra_smoothness_loss,
            boundary_smoothness_loss,
        )

    def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
        assert not self.residual_actor, "Not implemented"
        obs: dict[str, torch.Tensor] = batch["obs"]

        # Inject the projected (detached) task feature into observation.state
        # *in place* on batch["obs"], because the bc_loss_dynamic path in
        # update_actor_rft reads bc_batch.obs["observation.state"] and
        # bc_batch.obs["feat"] directly afterwards and expects them to match
        # the augmented prop_dim seen by the critic.
        if self.lang_enabled and obs["observation.state"].size(-1) == self._raw_prop_dim:
            key = self.lang_cfg.lang_emb_obs_key
            assert key in obs, (
                f"Language fusion enabled but bc obs is missing '{key}'."
            )
            with torch.no_grad():
                feat = self.lang_encoder(obs[key])
            state = obs["observation.state"]
            if state.dim() == 2 and feat.dim() == 2 and feat.size(0) == 1 and state.size(0) > 1:
                feat = feat.expand(state.size(0), -1)
            obs["observation.state"] = torch.cat([state, feat], dim=-1)

        assert "feat" not in obs, "safety check"
        obs["feat"] = self._encode(obs, augment=True)

        if not backprop_encoder:
            obs["feat"] = obs["feat"].detach()

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            clip=None,
            use_target=False,
        )
        action: torch.Tensor = batch["action"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss  # noqa: RET504

    def update_actor(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        prev_residual_last: torch.Tensor | None = None,
        has_prev_residual: torch.Tensor | None = None,
    ):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
            intra_smoothness_loss,
            boundary_smoothness_loss,
        ) = self._compute_actor_loss(
            obs, stddev, prev_residual_last, has_prev_residual
        )

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        metrics["train/actor_intra_chunk_smoothness_loss"] = intra_smoothness_loss.item()
        metrics["train/actor_boundary_smoothness_loss"] = boundary_smoothness_loss.item()
        metrics["train/actor_intra_chunk_smoothness_penalty"] = (
            self.cfg.temporal_smoothness_intra_weight * intra_smoothness_loss
        ).item()
        metrics["train/actor_boundary_smoothness_penalty"] = (
            self.cfg.temporal_smoothness_boundary_weight * boundary_smoothness_loss
        ).item()
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()

        # Log L2 regularization penalty if applied
        if self.cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        # Log learnable residual scale statistics if the actor exposes them
        sigma = getattr(self.actor, "last_sigma", None)
        if sigma is not None:
            sigma_det = sigma.detach()
            metrics["train/scale_sigma_mean"] = sigma_det.mean().item()
            metrics["train/scale_sigma_max"] = sigma_det.max().item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_total.backward()

        # Gradient clipping
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)

        # Store gradient norm for logging
        metrics["train/actor_grad_norm"] = actor_grad_norm.item()

        self.actor_opt.step()

        return metrics

    def update_actor_rft(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        bc_batch,
        ref_agent: QAgent,
        prev_residual_last: torch.Tensor | None = None,
        has_prev_residual: torch.Tensor | None = None,
    ):
        metrics = {}

        # Compute actor loss and get the actions used (single actor call)
        (
            actor_loss_total,
            actor_loss_base,
            combined_action,
            action_pred,
            action_l2_penalty,
            intra_smoothness_loss,
            boundary_smoothness_loss,
        ) = self._compute_actor_loss(
            obs, stddev, prev_residual_last, has_prev_residual
        )

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss_total.item()
        metrics["train/actor_intra_chunk_smoothness_loss"] = intra_smoothness_loss.item()
        metrics["train/actor_boundary_smoothness_loss"] = boundary_smoothness_loss.item()
        metrics["train/actor_intra_chunk_smoothness_penalty"] = (
            self.cfg.temporal_smoothness_intra_weight * intra_smoothness_loss
        ).item()
        metrics["train/actor_boundary_smoothness_penalty"] = (
            self.cfg.temporal_smoothness_boundary_weight * boundary_smoothness_loss
        ).item()
        # Store residual actions for logging (the actual residual component we want to monitor)
        metrics["_actions"] = action_pred.detach().cpu()
        # Also store combined actions if needed for other purposes
        metrics["_combined_actions"] = combined_action.detach().cpu()

        # Log L2 regularization penalty if applied
        if self.cfg.actor.action_l2_reg_weight > 0:
            metrics["train/actor_l2_penalty"] = action_l2_penalty.item()

        # Log learnable residual scale statistics if the actor exposes them
        sigma = getattr(self.actor, "last_sigma", None)
        if sigma is not None:
            sigma_det = sigma.detach()
            metrics["train/scale_sigma_mean"] = sigma_det.mean().item()
            metrics["train/scale_sigma_max"] = sigma_det.max().item()

        # Use config option to control whether BC loss updates encoder
        bc_backprop_encoder = self.cfg.bc_backprop_encoder
        bc_loss = self._compute_actor_bc_loss(bc_batch, backprop_encoder=bc_backprop_encoder)
        assert actor_loss_total.size() == bc_loss.size()

        ratio = 1
        if self.cfg.bc_loss_dynamic:
            with torch.no_grad(), utils.eval_mode(self, ref_agent):
                assert ref_agent.cfg.act_method == "rl"

                # temporarily change to rl since we want to regularize actor not hybrid
                act_method = self.cfg.act_method
                self.cfg.act_method = "rl"

                ref_bc_obs = bc_batch.obs.copy()  # shallow copy
                ref_action = ref_agent.act(ref_bc_obs, eval_mode=True, cpu=False)

                # we first get the ref_action and then pop the feature
                # then we get the curr_action so that the obs["feat"] is the current feature
                # which can be used for computing q-values
                bc_obs = bc_batch.obs
                curr_action = self.act(bc_obs, eval_mode=True, cpu=False)

                curr_q = self._critic_q_value_for_policy(bc_obs, curr_action)
                ref_q = self._critic_q_value_for_policy(bc_obs, ref_action)

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.cfg.act_method = act_method

        loss = actor_loss_total + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        # Conditionally update encoder along with actor if BC loss should backprop
        if bc_backprop_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        # Gradient clipping
        metrics["train/actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.actor_grad_clip_norm
        ).item()

        if bc_backprop_encoder:
            metrics["train/encoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                self.encoders.parameters(), self.cfg.actor_grad_clip_norm
            ).item()

        if bc_backprop_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        metrics["rft/bc_loss"] = bc_loss.mean().item()
        metrics["rft/ratio"] = ratio
        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
        bc_batch=None,
        ref_agent: "QAgentLang | None" = None,
    ):
        obs: dict[str, torch.Tensor] = batch["obs"]
        action: torch.Tensor = batch["action"]
        reward: torch.Tensor = batch[("next", "reward")]
        discount: torch.Tensor = batch["gamma"]
        next_nonterminal: torch.Tensor = batch["nonterminal"]
        next_obs: dict[str, torch.Tensor] = batch[("next", "obs")]

        # To not bootstrap on terminal states we zero out the discount factor for terminal next states
        effective_discount = discount * next_nonterminal

        # Critic phase: gradients flow through the language encoder so it
        # gets trained by the TD objective.
        obs = self._augment_state(obs, detach_lang=False)
        # next_obs is only used inside no_grad for target computation, so the
        # grad-flow flag has no effect there.
        next_obs = self._augment_state(next_obs, detach_lang=False)

        obs["feat"] = self._encode(obs, augment=True)

        with torch.no_grad():
            next_obs["feat"] = self._encode(next_obs, augment=True)

        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()

        # Extract importance sampling weights if available (for prioritized experience replay)
        importance_weights = batch.get("_weight", None)
        critic_metric = self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            discount=effective_discount,
            next_obs=next_obs,
            stddev=stddev,
            importance_weights=importance_weights,
        )
        self._soft_update_task_networks(self.critic, self.critic_target, obs)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        # NOTE: actor loss does not backprop into the image encoder, and we
        # also block gradients into the language encoder here (it has already
        # been trained via the critic objective above).
        if self.lang_enabled:
            # Re-augment the *original* (unaugmented) obs with a detached lang
            # feature.  We pull the un-augmented state from batch["obs"] which
            # has not been modified in place because _augment_state returned a
            # shallow copy.
            actor_obs = self._augment_state(batch["obs"], detach_lang=True)
            actor_obs["feat"] = obs["feat"].detach()
        else:
            actor_obs = obs
            actor_obs["feat"] = actor_obs["feat"].detach()

        prev_residual_last = batch.get("prev_residual_last", None)
        has_prev_residual = batch.get("has_prev_residual", None)
        if bc_batch is None:
            actor_metric = self.update_actor(
                actor_obs,
                stddev,
                prev_residual_last=prev_residual_last,
                has_prev_residual=has_prev_residual,
            )
        else:
            assert ref_agent is not None
            actor_metric = self.update_actor_rft(
                actor_obs,
                stddev,
                bc_batch,
                ref_agent,
                prev_residual_last=prev_residual_last,
                has_prev_residual=has_prev_residual,
            )

        self._soft_update_task_networks(self.actor, self.actor_target, actor_obs)
        metrics.update(actor_metric)

        return metrics

    def step_lr_schedulers(self):
        """Step the learning rate schedulers for warmup."""
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
