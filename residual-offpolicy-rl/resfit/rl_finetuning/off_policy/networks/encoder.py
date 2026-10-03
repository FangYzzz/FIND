# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import VitEncoderConfig, SiglipEncoderConfig
from resfit.rl_finetuning.off_policy.networks.min_vit import MinVit

from dataclasses import dataclass

import torch.nn.functional as F
from transformers import SiglipImageProcessor, SiglipVisionModel


class VitEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], cfg: VitEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def forward(self, obs, flatten=True) -> torch.Tensor:
        if obs.max() > 5:
            obs = obs / 255.0
        obs = obs - 0.5
        feats: torch.Tensor = self.vit.forward(obs)
        if flatten:
            # [B, D, N] -> [B, D*N]
            feats = feats.flatten(1, 2)
        return feats


if __name__ == "__main__":
    vit_encoder = VitEncoder(obs_shape=(3, 84, 84), cfg=VitEncoderConfig())
    obs = torch.rand(10, 3, 84, 84)
    feats: torch.Tensor = vit_encoder(obs)
    print(feats.size())  # (10, 10368), i.e., 81 patches * 128 dimensions


class SiglipEncoder(nn.Module):
    """
    Drop-in replacement for VitEncoder.

    Output convention:
    - flatten=False: [B, N, D]
    - flatten=True:  [B, N * D]

    This matches the current QAgent expectation where features are concatenated
    across cameras along the patch dimension.
    """

    def __init__(self, obs_shape: tuple[int, int, int], cfg: SiglipEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg

        self.processor = SiglipImageProcessor.from_pretrained(cfg.model_name)
        self.vision_model = SiglipVisionModel.from_pretrained(cfg.model_name)

        if cfg.freeze:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        vision_cfg = self.vision_model.config

        # Hidden size of each patch token
        self.patch_repr_dim = int(vision_cfg.hidden_size)

        # Number of visual tokens for the configured image size.
        # For ViT-like patching: (image_size // patch_size)^2
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            dummy = self._preprocess(dummy)
            feats = self.vision_model(pixel_values=dummy).last_hidden_state
            if cfg.drop_cls_token and feats.shape[1] > 1:
                feats = feats[:, 1:, :]
            self.num_patch = feats.shape[1]

        # If the model includes an extra token and you decide to keep it,
        # repr_dim can be updated dynamically in forward, but here we keep the
        # standard patch-token count.
        self.repr_dim = self.patch_repr_dim * self.num_patch

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Convert input to float and resize to SigLIP's expected resolution.

        Expected input:
        - uint8 [B, C, H, W] in [0, 255], or
        - float [B, C, H, W], typically already in [0, 1]
        """
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        else:
            obs = obs.float()

        # Resize to model resolution if needed
        target_size = self.cfg.force_image_size
        if obs.shape[-1] != target_size or obs.shape[-2] != target_size:
            obs = F.interpolate(
                obs,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        # Either rely on processor statistics, or do the minimal old-style centering.
        # Processor normalization is closer to the pretrained SigLIP recipe.
        if self.cfg.use_processor_norm:
            image_mean = torch.tensor(
                self.processor.image_mean,
                device=obs.device,
                dtype=obs.dtype,
            ).view(1, -1, 1, 1)
            image_std = torch.tensor(
                self.processor.image_std,
                device=obs.device,
                dtype=obs.dtype,
            ).view(1, -1, 1, 1)
            obs = (obs - image_mean) / image_std
        else:
            obs = obs - 0.5

        return obs

    def forward(self, obs: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        obs = self._preprocess(obs)

        outputs = self.vision_model(pixel_values=obs)
        feats = outputs.last_hidden_state  # [B, N, D]

        # Some vision transformers may include an extra token; keep behavior configurable.
        if self.cfg.drop_cls_token and feats.shape[1] == self.num_patch + 1:
            feats = feats[:, 1:, :]

        # Safety: if the actual token count differs from init-time estimate, update repr_dim.

        if flatten:
            feats = feats.flatten(1, 2)  # [B, N, D] -> [B, N*D]

        return feats
