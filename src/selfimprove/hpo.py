"""Hyperparameter fine-tune pass over the winning compact architecture.

A small random search around a centre config (the self-improve winner): resample
learning rate, width, depth and dropout, retrain centrally, and keep the model with
the best validation macro-F1 that still fits the size budget. Deliberately tiny (a handful
of trials) - this refines the winner, it does not re-run the whole search.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from config.base import Config
from src.data.preprocess import ProcessedData
from src.selfimprove.compress import state_dict_bytes
from src.training.centralized import train_centralized
from src.training.common import eval_macro_f1
from utils.logging_config import get_logger

_log = get_logger(__name__)


def hpo_finetune(
    center: Config,
    data: ProcessedData,
    device: str,
    n_trials: int = 5,
    size_budget_bytes: int = 2_000_000,
) -> dict[str, Any] | None:
    """Random-search fine-tune around ``center``. Returns the best trial or ``None``.

    Result dict: ``model``, ``cfg``, ``macro_f1``, ``size_bytes``, ``hidden_width``,
    ``hidden_layers``, ``dropout``, ``lr``. ``None`` when no trial fits the budget.
    """
    rng = np.random.default_rng(center.seed)
    w0, l0 = center.model.hidden_width, center.model.hidden_layers
    best: dict[str, Any] | None = None
    trials = 1 if center.runtime.fast else n_trials
    for _ in range(trials):
        width = int(np.clip(rng.integers(max(8, w0 // 2), w0 + 1), 8, w0))
        layers = int(rng.integers(1, max(1, l0) + 1))
        dropout = float(round(rng.uniform(0.0, 0.4), 2))
        lr = float(center.training.lr * rng.choice([0.3, 0.5, 1.0, 2.0, 3.0]))
        cfg = replace(
            center,
            model=replace(
                center.model, hidden_width=width, hidden_layers=layers, dropout=dropout
            ),
            training=replace(center.training, lr=lr),
        )
        model, _, m = train_centralized(data, cfg, device)
        size = state_dict_bytes(model)
        vf1 = eval_macro_f1(
            model, data.X_val, data.y_val, len(data.label_names), device
        )
        if size > size_budget_bytes:
            continue
        if best is None or vf1 > best["val_macro_f1"]:  # select on val, never test
            best = {
                "model": model,
                "cfg": cfg,
                "macro_f1": float(m["macro_f1"]),
                "val_macro_f1": vf1,
                "size_bytes": size,
                "hidden_width": width,
                "hidden_layers": layers,
                "dropout": dropout,
                "lr": lr,
            }
    if best is not None:
        _log.info(
            "HPO fine-tune: best F1=%.3f (%dw/%dL lr=%.2g size=%dB)",
            best["macro_f1"],
            best["hidden_width"],
            best["hidden_layers"],
            best["lr"],
            best["size_bytes"],
        )
    return best
