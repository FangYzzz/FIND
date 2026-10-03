# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""LanguageEncoder
================

Maps a per-step language signal (a precomputed sentence embedding) to a small
fixed-length task feature vector that can be concatenated with the proprio
state and fed into the actor / critic networks.

Convention
----------
Caller is expected to put a precomputed embedding in
``obs[cfg.lang_emb_obs_key]`` of shape ``[B, lang_emb_dim]`` (or ``[lang_emb_dim]``
which will be unsqueezed to ``[1, lang_emb_dim]``).  The embedding stays
constant within an episode (computed once from the task / prompt string with a
frozen text encoder such as CLIP / SBERT / T5).

A small MLP projects ``lang_emb_dim -> lang_proj_dim``; this projection is
trained jointly with the critic.
"""

from __future__ import annotations

import torch
from torch import nn


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        *,
        lang_emb_dim: int,
        lang_proj_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert lang_emb_dim > 0, "lang_emb_dim must be > 0"
        assert lang_proj_dim > 0, "lang_proj_dim must be > 0"
        assert num_layers >= 1, "num_layers must be >= 1"

        self.lang_emb_dim = lang_emb_dim
        self.lang_proj_dim = lang_proj_dim

        layers: list[nn.Module] = []
        in_dim = lang_emb_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, lang_proj_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(lang_proj_dim))

        self.proj = nn.Sequential(*layers)

    @property
    def out_dim(self) -> int:
        return self.lang_proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.proj(z)