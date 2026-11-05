"""SplittableMLP: view the shared MLP as (ClientNet, ServerNet) (05_MODELING §2).

The split is a VIEW over the same parameters, never a re-instantiation, so that
``chain(split(d)) is exactly the unsplit model``. ``tests/`` asserts this for
every ``d``; it is the gate that stops a subtly-wrong split from silently
invalidating every distributed result.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.mlp import SharedMLP


class ClientNet(nn.Module):
    """Hidden blocks 1..d, held locally by a client. Shares parent parameters."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks  # same module objects as the parent — not copies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ServerNet(nn.Module):
    """Hidden blocks d+1..L plus the head, held by the server. Shares parameters."""

    def __init__(self, blocks: nn.ModuleList, head: nn.Module) -> None:
        super().__init__()
        self.blocks = blocks
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class SplittableMLP(SharedMLP):
    """SharedMLP with a ``split(d)`` that returns parameter-sharing halves."""

    def split(self, d: int) -> tuple[ClientNet, ServerNet]:
        """Split after ``d`` client-side hidden layers.

        Args:
            d: Number of hidden layers assigned to the client (1..hidden_layers).

        Returns:
            ``(ClientNet, ServerNet)`` sharing this model's parameters.
        """
        n = len(self.blocks)
        if not 1 <= d <= n:
            raise ValueError(f"cut depth d={d} must be in 1..{n}")
        client = ClientNet(self.blocks[:d])
        server = ServerNet(self.blocks[d:], self.head)
        return client, server
