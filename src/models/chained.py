"""ChainedModel: reunite ClientNet + ServerNet as one nn.Module.

A real ``nn.Module`` (not a lambda) is required so SHAP's gradient explainers can
register hooks (03_ARCHITECTURE §4) and for the T1 white-box attack, which needs
to backpropagate through the whole model.
"""

from __future__ import annotations

import torch
from torch import nn


class ChainedModel(nn.Module):
    """Run ``server(client(x))`` as a single differentiable module."""

    # Explicit annotations so attribute access keeps the concrete type rather
    # than nn.Module.__getattr__'s ``Tensor | Module`` union.
    client: nn.Module
    server: nn.Module

    def __init__(self, client: nn.Module, server: nn.Module) -> None:
        super().__init__()
        self.client = client
        self.server = server

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.server(self.client(x))
