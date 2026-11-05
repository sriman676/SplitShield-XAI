"""Helpers of scripts/improve_flows.py: the keep/reject rule, metrics, fine-tuning and
constrained adversarial rows (owner 2026-10-15)."""

from __future__ import annotations

import numpy as np
import torch

from scripts import improve_flows as imf
from src.models.mlp import SharedMLP


def _m(p: float, r: float, fa: float) -> dict[str, float]:
    return {"precision": p, "recall": r, "false_alarm_rate": fa}


def test_not_worse_rule() -> None:
    old = _m(0.99, 0.95, 0.01)
    assert imf.not_worse(_m(0.99, 0.95, 0.01), old)
    assert imf.not_worse(_m(0.995, 0.96, 0.005), old)
    assert not imf.not_worse(_m(0.98, 0.99, 0.0), old)  # precision lower
    assert not imf.not_worse(_m(1.0, 0.94, 0.0), old)  # recall lower
    assert not imf.not_worse(_m(1.0, 1.0, 0.02), old)  # more false alarms


class _Fixed(torch.nn.Module):
    def __init__(self, logits: np.ndarray) -> None:
        super().__init__()
        self.logits = torch.as_tensor(logits, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits[x[:, 0].long()]


def test_det_metrics_counts() -> None:
    # rows: benign->benign, benign->attack (FA), attack->attack, attack->benign (miss)
    logits = np.array([[5, 0], [0, 5], [0, 5], [5, 0]], dtype=np.float32)
    X = np.arange(4, dtype=np.float32)[:, None]
    y = np.array([0, 0, 1, 1])
    m = imf.det_metrics(_Fixed(logits), X, y, benign=0)
    assert (m["caught"], m["false_alarms"]) == (1, 1)
    assert m["recall"] == 0.5 and m["false_alarm_rate"] == 0.5 and m["precision"] == 0.5


def test_finetune_keeps_best_validation_epoch() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    mlp = SharedMLP(
        n_features=4, n_classes=2, hidden_layers=1, hidden_width=8, dropout=0.0
    )
    before = imf._macro_f1(mlp, X[400:], y[400:])
    tuned, hist = imf.finetune(
        mlp, X[:400], y[:400], X[400:], y[400:], epochs=5, lr=1e-2, batch=64
    )
    assert imf._macro_f1(tuned, X[400:], y[400:]) == max(hist) >= before
    assert tuned is not mlp  # the champion itself is never modified


def test_pgd_rows_respect_constraints() -> None:
    torch.manual_seed(0)
    mlp = SharedMLP(
        n_features=3, n_classes=2, hidden_layers=1, hidden_width=8, dropout=0.0
    ).eval()
    x = torch.zeros(20, 3)
    y = torch.ones(20, dtype=torch.long)
    mask = torch.tensor([1.0, 1.0, 0.0])  # third feature frozen
    inc = torch.tensor([True, False, False])  # first may only grow
    xa = imf._pgd_rows(mlp, x, y, mask, inc, eps=0.1)
    assert torch.all(xa[:, 2] == 0)
    assert torch.all(xa[:, 0] >= 0)
    assert torch.all((xa - x).abs() <= 0.1 + 1e-6)
