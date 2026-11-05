"""Generator + Discriminator for the T2 GAN evasion attack (03_ARCHITECTURE §1).

Small MLP GAN that models a source attack class in scaled feature space. The
target IDS is NOT part of training (T2 is black-box, 02_THREAT_MODEL §2); it is
queried as a label oracle only, in ``src/attack/gan_attack.py``.
"""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """noise -> scaled feature vector."""

    def __init__(self, noise_dim: int, n_features: int, hidden: int = 128) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """scaled feature vector -> real/fake logit."""

    def __init__(self, n_features: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
