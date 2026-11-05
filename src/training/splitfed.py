"""A3 — SplitFed (Thapa et al. 2022).

Clients train in parallel against their own ClientNet + ServerNet replica; both
halves are averaged each round. This is the arm that should NOT catastrophically
forget, which is what makes A1's sequential behaviour a finding rather than a bug.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from config.base import Config
from src.data.partition import partition_dirichlet
from src.data.preprocess import ProcessedData
from src.eval.metrics import classification_metrics
from src.models.chained import ChainedModel
from src.models.splittable import SplittableMLP
from src.training.centralized import predict
from src.training.common import (
    EarlyStopper,
    average_state_dicts,
    eval_macro_f1,
    make_loader,
)


def _build(data: ProcessedData, cfg: Config, device: str) -> SplittableMLP:
    return SplittableMLP(
        n_features=len(data.feature_names),
        n_classes=len(data.label_names),
        hidden_layers=cfg.model.hidden_layers,
        hidden_width=cfg.model.hidden_width,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
    ).to(device)


def _local_split_train(
    model: SplittableMLP, X, y, cfg: Config, device: str, criterion: nn.Module
) -> None:
    """One client's local split-style training pass, in place."""
    client, server = model.split(cfg.model.cut_depth)
    opt_c = torch.optim.AdamW(
        client.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    opt_s = torch.optim.AdamW(
        server.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    loader = make_loader(X, y, cfg.runtime.batch_size, cfg.seed, cfg.runtime)
    for _ in range(cfg.training.local_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z = client(xb)
            z_server = z.detach().requires_grad_(True)
            loss = criterion(server(z_server), yb)
            opt_c.zero_grad()
            opt_s.zero_grad()
            loss.backward()
            z.backward(z_server.grad)
            opt_s.step()
            opt_c.step()


def train_splitfed(
    data: ProcessedData, cfg: Config, device: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Train A3 (SplitFed) and evaluate once on test."""
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
            _local_split_train(
                local, data.X_train[cidx], data.y_train[cidx], cfg, device, criterion
            )
            states.append(local.state_dict())
            sizes.append(int(len(cidx)))
        global_model.load_state_dict(average_state_dicts(states, sizes))
        chained = ChainedModel(*global_model.split(cfg.model.cut_depth))
        vf1 = eval_macro_f1(chained, data.X_val, data.y_val, n_classes, device)
        history.append(vf1)
        if stopper.step(vf1, global_model):
            break
    stopper.restore(global_model)

    chained = ChainedModel(*global_model.split(cfg.model.cut_depth))
    metrics = classification_metrics(
        data.y_test, predict(chained, data.X_test, device), data.label_names
    )
    return chained, {"per_round_val_macro_f1": history, "test": metrics}
