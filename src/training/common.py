"""Shared training-loop scaffolding: dataloaders and early stopping.

Kept minimal and imported by every training scheme so the split-learning and
FedAvg loops (Task 3) reuse the same batching and stopping discipline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from config.base import RuntimeConfig
from utils.seed import seeded_loader_kwargs


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    runtime: RuntimeConfig,
    shuffle: bool = True,
) -> DataLoader:
    """Build a reproducible DataLoader over in-memory arrays."""
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    kwargs: dict[str, Any] = seeded_loader_kwargs(seed) if shuffle else {}
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=runtime.num_workers,
        pin_memory=runtime.pin_memory,
        **kwargs,
    )


class EarlyStopper:
    """Early stopping on a maximised validation metric (macro-F1)."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best: float = -1.0
        self.best_state: dict | None = None
        self._since = 0

    def step(self, metric: float, model: torch.nn.Module) -> bool:
        """Record a metric; return True when training should stop."""
        if metric > self.best:
            self.best = metric
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self._since = 0
            return False
        self._since += 1
        return self._since >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        """Load the best-seen weights back into ``model``."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


@torch.no_grad()
def eval_macro_f1(
    model: torch.nn.Module, X: np.ndarray, y: np.ndarray, n_classes: int, device: str
) -> float:
    """Macro-F1 of ``model`` on ``(X, y)`` over the fixed label set."""
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
    pred = logits.argmax(dim=1).cpu().numpy()
    return float(
        f1_score(
            y, pred, labels=list(range(n_classes)), average="macro", zero_division=0
        )
    )


def average_state_dicts(states: list[dict], weights: list[int]) -> dict:
    """Sample-count-weighted average of state dicts (FedAvg / SplitFed)."""
    total = float(sum(weights))
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in states[0].items()}
    for state, w in zip(states, weights, strict=False):
        for k, v in state.items():
            avg[k] += v.float() * (w / total)
    return {k: v.to(states[0][k].dtype) for k, v in avg.items()}
