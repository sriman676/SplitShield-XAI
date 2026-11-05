"""The single shared MLP architecture (05_MODELING §2).

One definition, used by every arm in Stages 2-8. Fixed at 4 hidden layers so the
cut-depth ablation has meaningful cuts d in {1,2,3}. Hidden blocks are kept in a
``ModuleList`` so ``splittable.py`` (Task 3) can view them as ClientNet/ServerNet
without re-instantiating parameters.

BatchNorm default off (05_MODELING §2): running stats crossing the cut boundary
would weaken the privacy story and upset some SHAP explainers.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(in_dim: int, out_dim: int, dropout: float, norm: str) -> nn.Sequential:
    """One hidden block: Linear -> (norm?) -> ReLU -> Dropout."""
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if norm == "layer":
        layers.append(nn.LayerNorm(out_dim))
    elif norm == "batch":
        layers.append(nn.BatchNorm1d(out_dim))
    layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class SharedMLP(nn.Module):
    """4-hidden-layer MLP over tabular flow features -> 9-class logits."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 9,
        hidden_layers: int = 4,
        hidden_width: int = 128,
        dropout: float = 0.2,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _block(n_features if i == 0 else hidden_width, hidden_width, dropout, norm)
            for i in range(hidden_layers)
        )
        self.head = nn.Linear(hidden_width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all hidden blocks then the classification head."""
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def param_count(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
