"""A1 — vanilla sequential split learning (03_ARCHITECTURE §2 Stage 2).

One ClientNet is carried between clients in turn; the server holds one ServerNet.
Per turn: client forward -> detached cut activations -> server forward/backward ->
gradient returned at the cut -> client backward. Evaluated at END OF CYCLE, with
per-client-turn macro-F1 logged so P4's oscillation is visible (figure F5).

Under severe skew this is EXPECTED to look bad (last-client bias). Do not lower
the LR to hide it — that erases the finding (05_MODELING §5).
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
from src.training.common import EarlyStopper, eval_macro_f1, make_loader
from utils.logging_config import get_logger

_log = get_logger(__name__)


def _build(data: ProcessedData, cfg: Config, device: str) -> SplittableMLP:
    return SplittableMLP(
        n_features=len(data.feature_names),
        n_classes=len(data.label_names),
        hidden_layers=cfg.model.hidden_layers,
        hidden_width=cfg.model.hidden_width,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
    ).to(device)


def train_split_sequential(
    data: ProcessedData, cfg: Config, device: str = "cpu"
) -> tuple[ChainedModel, dict[str, Any]]:
    """Train A1 and evaluate once on test.

    Returns:
        ``(chained_model, result)`` where ``result`` holds per-cycle and
        per-client-turn val macro-F1 histories plus the test metrics.
    """
    model = _build(data, cfg, device)
    client, server = model.split(cfg.model.cut_depth)
    chained = ChainedModel(client, server)
    n_classes = len(data.label_names)

    weight = (
        torch.as_tensor(data.class_weights, dtype=torch.float32, device=device)
        if data.class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt_c = torch.optim.AdamW(
        client.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    opt_s = torch.optim.AdamW(
        server.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    parts = partition_dirichlet(
        data.y_train, cfg.training.clients, cfg.training.dirichlet_alpha, cfg.seed
    )
    stopper = EarlyStopper(cfg.training.patience)

    per_cycle: list[float] = []
    per_turn: list[float] = []
    for _cycle in range(cfg.training.sl_cycles):
        for cidx in parts:
            if len(cidx) == 0:
                continue
            loader = make_loader(
                data.X_train[cidx],
                data.y_train[cidx],
                cfg.runtime.batch_size,
                cfg.seed,
                cfg.runtime,
            )
            for _ in range(cfg.training.local_epochs):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    z = client(xb)
                    z_server = z.detach().requires_grad_(True)
                    loss = criterion(server(z_server), yb)
                    opt_c.zero_grad()
                    opt_s.zero_grad()
                    loss.backward()  # server grads + z_server.grad at the cut
                    z.backward(z_server.grad)  # client half
                    opt_s.step()
                    opt_c.step()
            per_turn.append(
                eval_macro_f1(chained, data.X_val, data.y_val, n_classes, device)
            )
        vf1 = eval_macro_f1(chained, data.X_val, data.y_val, n_classes, device)
        per_cycle.append(vf1)
        if stopper.step(vf1, model):
            break
    stopper.restore(model)

    metrics = classification_metrics(
        data.y_test, predict(chained, data.X_test, device), data.label_names
    )
    return chained, {
        "per_cycle_val_macro_f1": per_cycle,
        "per_turn_val_macro_f1": per_turn,
        "test": metrics,
    }
