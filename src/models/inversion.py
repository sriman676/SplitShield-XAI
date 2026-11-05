"""Inversion network for T3 activation inference (03_ARCHITECTURE §1).

A small MLP that maps intercepted cut-layer activations back to input features.
Trained by the honest-but-curious server on auxiliary same-distribution data; see
``src/attack/activation_inference.py`` for the threat-model framing and the
load-bearing auxiliary-data assumption.
"""

from __future__ import annotations

import torch
from torch import nn


class InversionNet(nn.Module):
    """activation_dim -> hidden -> n_features regression head."""

    def __init__(self, activation_dim: int, n_features: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(activation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)
