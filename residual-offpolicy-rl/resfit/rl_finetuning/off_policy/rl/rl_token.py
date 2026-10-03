"""RL Token bottleneck used to compress frozen VLA vision-language features."""

from __future__ import annotations

import torch
from torch import nn


class RLTokenVAE(nn.Module):
    """Transformer VAE with a learned readout token and sequence decoder.

    Input is the final VLA prefix sequence ``[B, N, D]``.  Padding positions may
    be supplied through ``padding_mask`` (True means padding).
    """

    def __init__(self, input_dim: int, token_dim: int, model_dim: int, num_heads: int,
                 encoder_layers: int, decoder_layers: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.token_dim = token_dim
        # pi0 hidden states can have a large, checkpoint-dependent scale.
        # Normalize each token before both encoding and reconstruction so the
        # VAE objective and KL term remain well-conditioned.
        self.embedding_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.rl_query = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            model_dim, num_heads, model_dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, encoder_layers)
        self.to_mu = nn.Linear(model_dim, token_dim)
        self.to_logvar = nn.Linear(model_dim, token_dim)
        self.from_token = nn.Linear(token_dim, model_dim)
        self.decoder_queries = nn.Parameter(torch.randn(1, 2048, model_dim) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            model_dim, num_heads, model_dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, decoder_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)

    def encode_distribution(self, embeddings, padding_mask=None):
        x = self.input_proj(self.embedding_norm(embeddings.float()))
        query = self.rl_query.expand(x.shape[0], -1, -1)
        x = torch.cat((query, x), dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat(
                (torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), padding_mask), dim=1
            )
        readout = self.encoder(x, src_key_padding_mask=padding_mask)[:, 0]
        return self.to_mu(readout), self.to_logvar(readout).clamp(-10.0, 10.0)

    def encode(self, embeddings, padding_mask=None):
        """Deterministic frozen feature used by actor and critic."""
        return self.encode_distribution(embeddings, padding_mask)[0]

    def forward(self, embeddings, padding_mask=None):
        mu, logvar = self.encode_distribution(embeddings, padding_mask)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        memory = self.from_token(z).unsqueeze(1)
        if embeddings.shape[1] > self.decoder_queries.shape[1]:
            raise ValueError(f"VLA sequence length {embeddings.shape[1]} exceeds decoder limit 2048")
        # Learned position queries prevent a direct identity path from the
        # reconstruction target; all content must pass through the RL token.
        target = self.decoder_queries[:, :embeddings.shape[1]].expand(embeddings.shape[0], -1, -1)
        reconstruction = self.output_proj(self.decoder(target, memory))
        return reconstruction, mu, logvar

    def loss(self, reconstruction, target, mu, logvar, valid_mask=None, beta_kl=1e-4):
        normalized_target = self.embedding_norm(target.float())
        error = (reconstruction - normalized_target).square().mean(dim=-1)
        if valid_mask is not None:
            reconstruction_loss = (error * valid_mask).sum() / valid_mask.sum().clamp_min(1)
        else:
            reconstruction_loss = error.mean()
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean(dim=-1).mean()
        return reconstruction_loss + beta_kl * kl, reconstruction_loss, kl
