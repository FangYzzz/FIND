"""QAgent variant whose visual-language observation is a frozen RL token."""

from __future__ import annotations

import torch
from torch import nn

from resfit.rl_finetuning.off_policy.rl.q_agent_lang import QAgentLang


class _TokenEncoder(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self._optimizer_anchor = nn.Parameter(torch.zeros(()))
        self.repr_dim = token_dim
        self.patch_repr_dim = token_dim

    def forward(self, token: torch.Tensor, flatten: bool = False):
        del flatten
        if token.ndim == 1:
            token = token.unsqueeze(0)
        return token.float().unsqueeze(1) + self._optimizer_anchor * 0.0


class QAgentRLToken(QAgentLang):
    """Preserves TD3 logic while replacing camera/language encoders."""

    def __init__(self, *args, token_dim: int, token_obs_key: str, **kwargs):
        self._token_dim = token_dim
        kwargs["rl_cameras"] = [token_obs_key]
        kwargs["obs_shape"] = (token_dim, 1, 1)
        # Language is already fused inside the VLA token.
        if kwargs["cfg"].language is not None:
            kwargs["cfg"].language.enabled = False
        super().__init__(*args, **kwargs)

    def _build_encoders(self, obs_shape):
        del obs_shape
        return nn.ModuleList([_TokenEncoder(self._token_dim).to(self.cfg.device)])

    def _maybe_unsqueeze_(self, obs):
        key = self.rl_cameras[0]
        if obs[key].ndim == 1:
            for name, value in obs.items():
                obs[name] = value.unsqueeze(0)
            return True
        return False

    def _encode(self, obs, augment):
        del augment
        return self.encoders[0](obs[self.rl_cameras[0]], flatten=False)
