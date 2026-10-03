# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Drop-in replacement for Actor with a state-conditioned, learnable residual scale.

Instead of `scaled_mu = mu * cfg.action_scale` (a fixed constant), this actor
produces a per-state, per-dim gate `sigma(s) in (0, s_max]` and uses
`scaled_mu = sigma(s) * mu`. The base residual magnitude is no longer hand-tuned;
it is learned from the critic.

The scale shares the same trunk *and* the same final linear as the policy: the
final linear outputs `action_dim + sigma_dim` values, the first `action_dim`
slice is the action mean (via tanh), the remaining slice is the scale logits
(via `s_max * sigmoid(...)`).

External API matches `resfit.rl_finetuning.off_policy.rl.actor.Actor`:
- constructor signature is identical
- `forward(obs, std)` returns a `TruncatedNormal`

Optional ActorConfig fields (read via getattr so the existing config still works):
- scale_head_enabled        (bool,  default True): turn the learnable gate on
- scale_head_max            (float, default cfg.action_scale): hard cap for sigma
- scale_head_init_value     (float, default cfg.action_scale): initial sigma at step 0
- scale_head_per_dim        (bool,  default True): per-dim sigma (False = scalar gate)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils


def build_fc(in_dim, hidden_dim, out_dim, num_layer, layer_norm, dropout, use_layer_norm=True, final_activation=None):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm and layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if use_layer_norm and layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], out_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout, use_layer_norm=True):
        super().__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim

        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        layers = [nn.Linear(proj_in_dim, proj_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(proj_dim))
        layers.append(nn.ReLU(inplace=True))

        self.input_proj = nn.Sequential(*layers)
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2)

        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z


class Actor(nn.Module):
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: ActorConfig, residual_actor: bool = False):
        super().__init__()

        self.prop_dim = prop_dim
        self.residual_actor = residual_actor
        self.cfg = cfg
        self.action_dim = action_dim

        if residual_actor:
            self.prop_dim += action_dim

        if cfg.spatial_emb > 0:
            assert cfg.spatial_emb > 1, "this is the dimension"
            self.compress = SpatialEmb(
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                prop_dim=self.prop_dim,
                proj_dim=cfg.spatial_emb,
                dropout=cfg.dropout,
                use_layer_norm=cfg.use_layer_norm,
            )
            policy_in_dim = cfg.spatial_emb
        else:
            layers = [nn.Linear(repr_dim, cfg.feature_dim)]
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(cfg.feature_dim))
            layers.extend([nn.Dropout(cfg.dropout), nn.ReLU()])

            self.compress = nn.Sequential(*layers)
            policy_in_dim = cfg.feature_dim

        if self.prop_dim > 0:
            policy_in_dim += self.prop_dim

        self.scale_head_enabled: bool = bool(getattr(cfg, "scale_head_enabled", True))
        if self.scale_head_enabled:
            self.s_max = float(getattr(cfg, "scale_head_max", cfg.action_scale))
            init_value = float(getattr(cfg, "scale_head_init_value", cfg.action_scale))
            assert 0.0 < init_value < self.s_max, (
                f"scale_head_init_value ({init_value}) must satisfy 0 < init < s_max ({self.s_max})"
            )
            self._scale_per_dim = bool(getattr(cfg, "scale_head_per_dim", True))
            self._sigma_dim = action_dim if self._scale_per_dim else 1
            # logit(init_value / s_max) so sigma(s) starts near init_value
            p0 = init_value / self.s_max
            self._sigma_bias_init = math.log(p0 / (1.0 - p0))
        else:
            self.s_max = None
            self._scale_per_dim = False
            self._sigma_dim = 0
            self._sigma_bias_init = 0.0

        # Single policy MLP whose final layer outputs (mu | sigma_logits).
        # No terminal activation — we apply tanh / sigmoid per-slice in forward().
        policy_out_dim = action_dim + self._sigma_dim
        self.policy = build_fc(
            policy_in_dim,
            cfg.hidden_dim,
            policy_out_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
            final_activation=None,
        )

        # Cache last sigma for logging.
        self.last_sigma: torch.Tensor | None = None

        self._initialize_weights(cfg)

    def _initialize_weights(self, cfg: ActorConfig):
        intermediate_init = cfg.actor_intermediate_layer_init_distribution
        if cfg.orth and intermediate_init == "default":
            intermediate_init = "orthogonal"

        if cfg.orth:
            self.compress.apply(utils.orth_weight_init)
        else:
            utils.apply_initialization_to_network(self.compress, intermediate_init)

        utils.apply_initialization_to_network(self.policy, intermediate_init, exclude_final_layer=True)

        final_layer = self._find_final_linear(self.policy)

        if cfg.actor_last_layer_init_scale is not None and final_layer is not None:
            utils.initialize_layer_weights(
                final_layer,
                cfg.actor_last_layer_init_distribution,
                cfg.actor_last_layer_init_scale,
            )

        # Override the sigma rows of the final linear so sigma(s) starts near
        # init_value (zero weights + logit(init_value / s_max) bias).
        if self.scale_head_enabled and final_layer is not None:
            with torch.no_grad():
                final_layer.weight[self.action_dim :, :].zero_()
                final_layer.bias[self.action_dim :].fill_(self._sigma_bias_init)

    @staticmethod
    def _find_final_linear(module: nn.Module) -> nn.Linear | None:
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Linear):
                return m
        return None

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        if isinstance(self.compress, SpatialEmb):
            assert not self.residual_actor, "Not implemented"
            feat = self.compress.forward(obs["feat"], obs["observation.state"])
        else:
            feat = obs["feat"].flatten(1, -1)
            feat = self.compress(feat)

        all_input = [feat]
        if self.prop_dim > 0:
            prop = obs["observation.state"]
            all_input.append(prop)
            if self.residual_actor:
                all_input.append(obs["observation.base_action"])

        policy_input = torch.cat(all_input, dim=-1)

        out: torch.Tensor = self.policy(policy_input)  # (B, action_dim + sigma_dim)
        mu = torch.tanh(out[..., : self.action_dim])

        if self.scale_head_enabled:
            sigma_logits = out[..., self.action_dim :]
            sigma = self.s_max * torch.sigmoid(sigma_logits)
            if not self._scale_per_dim:
                sigma = sigma.expand(-1, self.action_dim)
            scaled_mu = sigma * mu
            self.last_sigma = sigma
        else:
            scaled_mu = mu * self.cfg.action_scale
            self.last_sigma = None

        action_dist = utils.TruncatedNormal(scaled_mu, std)
        return action_dist
