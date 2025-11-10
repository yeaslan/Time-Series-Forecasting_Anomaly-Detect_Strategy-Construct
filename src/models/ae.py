from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class FeedForwardAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.extend([nn.Linear(last_dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
            last_dim = hidden
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        last_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(last_dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
            last_dim = hidden
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
