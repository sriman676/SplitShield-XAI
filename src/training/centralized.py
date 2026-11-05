"""Centralized training (A0) — the upper-bound control for every distributed arm.

Trains the shared MLP on the union of data with class-weighted cross-entropy,
early-stopping on validation macro-F1. Test data is touched once, at the end
(05_MODELING §5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from config.base import Config
from src.data.preprocess import ProcessedData
from src.eval.metrics import classification_metrics
from src.models.mlp import SharedMLP, param_count
from src.training.common import EarlyStopper, make_loader
from utils.logging_config import get_logger

_log = get_logger(__name__)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Return argmax class predictions for ``X``."""
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
    return logits.argmax(dim=1).cpu().numpy()


def _macro_f1(
    model: nn.Module, X: np.ndarray, y: np.ndarray, data: ProcessedData, device: str
) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y,
            predict(model, X, device),
            average="macro",
            labels=list(range(len(data.label_names))),
            zero_division=0,
        )
    )


@torch.no_grad()
def _val_f1_loss(
    model: nn.Module, data: ProcessedData, device: str, criterion: nn.Module
) -> tuple[float, float]:
    """Validation macro-F1 and loss from ONE forward pass."""
    from sklearn.metrics import f1_score

    model.eval()
    logits = model(torch.as_tensor(data.X_val, dtype=torch.float32, device=device))
    y = torch.as_tensor(data.y_val, dtype=torch.long, device=device)
    f1 = f1_score(
        data.y_val,
        logits.argmax(1).cpu().numpy(),
        average="macro",
        labels=list(range(len(data.label_names))),
        zero_division=0,
    )
    return float(f1), float(criterion(logits, y))


def train_centralized(
    data: ProcessedData,
    cfg: Config,
    device: str = "cpu",
    init_state: dict[str, Any] | None = None,
) -> tuple[nn.Module, list[float], dict[str, Any]]:
    """Train the shared MLP centrally and evaluate once on test.

    ``init_state`` optionally warm-starts the model (e.g. from unsupervised
    pretraining); ``None`` (default) trains from scratch, the original behaviour.

    Returns:
        ``(model, val_macro_f1_history, test_metrics)``.
    """
    model = SharedMLP(
        n_features=len(data.feature_names),
        n_classes=len(data.label_names),
        hidden_layers=cfg.model.hidden_layers,
        hidden_width=cfg.model.hidden_width,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
    ).to(device)
    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
    _log.info(
        "Centralized MLP: %d params, %d features",
        param_count(model),
        len(data.feature_names),
    )

    weight = (
        torch.as_tensor(data.class_weights, dtype=torch.float32, device=device)
        if data.class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.training.max_epochs
    )
    loader = make_loader(
        data.X_train, data.y_train, cfg.runtime.batch_size, cfg.seed, cfg.runtime
    )
    stopper = EarlyStopper(cfg.training.patience)

    history: list[float] = []
    fit_hist: list[dict[str, float]] = []  # per-epoch val F1 + loss for the fit check
    stopped_early = False
    for epoch in range(cfg.training.max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        vf1, vloss = _val_f1_loss(model, data, device, criterion)
        history.append(vf1)
        fit_hist.append({"val_f1": vf1, "val_loss": vloss})
        if stopper.step(vf1, model):
            stopped_early = True
            _log.info(
                "Early stop at epoch %d (best val macro-F1=%.3f)", epoch, stopper.best
            )
            break
    stopper.restore(model)

    test_pred = predict(model, data.X_test, device)
    metrics = classification_metrics(data.y_test, test_pred, data.label_names)
    from src.eval.fitcheck import diagnose

    train_f1 = _macro_f1(model, data.X_train, data.y_train, data, device)
    metrics["fit"] = diagnose(
        fit_hist, train_f1, cfg.training.max_epochs, stopped_early
    )
    return model, history, metrics
