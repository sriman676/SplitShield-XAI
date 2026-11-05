"""A2 — FedAvg (McMahan et al. 2017). Non-split federated baseline.

Each client trains a full local copy of the shared MLP for ``local_epochs``; the
server averages weights by sample count each round. The comparison that shows
whether A1's sequential degradation is intrinsic or just a training-scheme choice.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from config.base import Config
from src.data.partition import partition_dirichlet
from src.data.preprocess import ProcessedData
from src.eval.metrics import classification_metrics
from src.models.mlp import SharedMLP
from src.training.centralized import predict
from src.training.common import (
    EarlyStopper,
    average_state_dicts,
    eval_macro_f1,
    make_loader,
)


def _build(data: ProcessedData, cfg: Config, device: str) -> SharedMLP:
    return SharedMLP(
        n_features=len(data.feature_names),
        n_classes=len(data.label_names),
        hidden_layers=cfg.model.hidden_layers,
        hidden_width=cfg.model.hidden_width,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
    ).to(device)


def train_fedavg(
    data: ProcessedData, cfg: Config, device: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Train A2 (FedAvg) and evaluate once on test."""
    global_model = _build(data, cfg, device)
    n_classes = len(data.label_names)
    weight = (
        torch.as_tensor(data.class_weights, dtype=torch.float32, device=device)
        if data.class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    parts = partition_dirichlet(
        data.y_train, cfg.training.clients, cfg.training.dirichlet_alpha, cfg.seed
    )
    stopper = EarlyStopper(cfg.training.patience)

    history: list[float] = []
    for _round in range(cfg.training.sl_cycles):
        states: list[dict] = []
        sizes: list[int] = []
        gstate = global_model.state_dict()
        for cidx in parts:
            if len(cidx) == 0:
                continue
            local = _build(data, cfg, device)
            local.load_state_dict(gstate)
            opt = torch.optim.AdamW(
                local.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
            loader = make_loader(
                data.X_train[cidx],
                data.y_train[cidx],
                cfg.runtime.batch_size,
                cfg.seed,
                cfg.runtime,
            )
            for _ in range(cfg.training.local_epochs):
                local.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    criterion(local(xb), yb).backward()
                    opt.step()
            states.append(local.state_dict())
            sizes.append(int(len(cidx)))
        global_model.load_state_dict(average_state_dicts(states, sizes))
        vf1 = eval_macro_f1(global_model, data.X_val, data.y_val, n_classes, device)
        history.append(vf1)
        if stopper.step(vf1, global_model):
            break
    stopper.restore(global_model)

    metrics = classification_metrics(
        data.y_test, predict(global_model, data.X_test, device), data.label_names
    )
    return global_model, {"per_round_val_macro_f1": history, "test": metrics}
