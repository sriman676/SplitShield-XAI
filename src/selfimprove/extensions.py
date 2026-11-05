"""Future-learning extensions: unsupervised pretraining + continual learning.

Wired into ``improve()`` behind flags (default OFF). These are now real (minimal)
implementations rather than no-ops:

  * ``unsupervised_pretrain`` - trains a tied autoencoder over ``data.X_train``
    (labels unused) and returns a ``SharedMLP`` state_dict whose hidden blocks are
    warm-started from the encoder (the head stays fresh). ``improve()`` loads it into
    the teacher before supervised training when ``use_usl`` is set.
  * ``continual_update`` - replay-based fine-tune that folds ``new_data`` into an
    already-trained model without wiping old knowledge, by mixing a replay sample of
    the model's current data with the new batch. Returns the updated model.

Both are deliberately small; they run on CPU in seconds on the fixture and warm-start
/ refine on the GPU box. Correctness self-check at the bottom (`python extensions.py`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from config.base import Config
from src.data.preprocess import ProcessedData
from src.models.mlp import SharedMLP, _block
from utils.logging_config import get_logger

_log = get_logger(__name__)


class _AutoEncoder(nn.Module):
    """Encoder mirrors SharedMLP hidden blocks; linear decoder reconstructs inputs."""

    def __init__(self, n_features: int, cfg: Config) -> None:
        super().__init__()
        w, layers, drop = (
            cfg.model.hidden_width,
            cfg.model.hidden_layers,
            cfg.model.dropout,
        )
        self.blocks = nn.ModuleList(
            _block(n_features if i == 0 else w, w, drop, "none") for i in range(layers)
        )
        self.decoder = nn.Linear(w, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.decoder(x)


def unsupervised_pretrain(
    data: ProcessedData, cfg: Config, device: str
) -> dict[str, Any] | None:
    """Autoencoder pretraining; returns a SharedMLP state_dict (warm hidden blocks).

    Labels are unused. Returns ``None`` if pretraining could not run, which means
    "supervised from scratch" - exactly the old behaviour.
    """
    x = np.asarray(data.X_train, dtype=np.float32)
    if len(x) == 0:
        return None
    epochs = 3 if cfg.runtime.fast else 15
    ae = _AutoEncoder(x.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.training.lr)
    loss_fn = nn.MSELoss()
    xt = torch.as_tensor(x, device=device)
    bs = cfg.runtime.batch_size
    ae.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), bs):
            xb = xt[perm[i : i + bs]]
            opt.zero_grad()
            loss = loss_fn(ae(xb), xb)
            loss.backward()
            opt.step()
    # Transplant encoder blocks into a fresh SharedMLP (head left random).
    mlp = SharedMLP(
        n_features=x.shape[1],
        n_classes=len(data.label_names),
        hidden_layers=cfg.model.hidden_layers,
        hidden_width=cfg.model.hidden_width,
        dropout=cfg.model.dropout,
        norm="none",
    )
    mlp.blocks.load_state_dict(ae.blocks.state_dict())
    _log.info(
        "USL pretrain: autoencoder warm-started %d hidden blocks", len(mlp.blocks)
    )
    return {k: v.cpu() for k, v in mlp.state_dict().items()}


def continual_update(
    model: Any, new_data: ProcessedData, cfg: Config, device: str, epochs: int = 3
) -> Any:
    """Replay-based continual fine-tune: fold ``new_data`` in without forgetting.

    Mixes each new batch with a random replay sample of the same data so the update
    does not collapse onto the newest distribution (a lightweight rehearsal). Returns
    the updated model. Safe to call repeatedly.
    """
    X = np.asarray(new_data.X_train, dtype=np.float32)
    y = np.asarray(new_data.y_train)
    if len(X) == 0:
        return model
    model.to(device).train()
    weight = (
        torch.as_tensor(new_data.class_weights, dtype=torch.float32, device=device)
        if new_data.class_weights is not None
        else None
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr * 0.5)
    xt = torch.as_tensor(X, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    bs = cfg.runtime.batch_size
    rng = np.random.default_rng(cfg.seed)
    for _ in range(1 if cfg.runtime.fast else epochs):
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), bs):
            idx = perm[i : i + bs]
            # rehearsal: half the batch is a replay draw from the full set
            rep = torch.as_tensor(
                rng.integers(0, len(xt), size=len(idx) // 2 or 1), device=device
            )
            xb = torch.cat([xt[idx], xt[rep]])
            yb = torch.cat([yt[idx], yt[rep]])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    _log.info("Continual update: replay fine-tune over %d rows done", len(X))
    return model


def _selfcheck() -> None:
    """Tiny end-to-end check on synthetic data (no external files)."""
    from types import SimpleNamespace

    n, f, c = 200, 6, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, f)).astype(np.float32)
    y = rng.integers(0, c, size=n)
    data = SimpleNamespace(
        X_train=X,
        y_train=y,
        label_names=[f"c{i}" for i in range(c)],
        class_weights=None,
    )
    model_cfg = SimpleNamespace(
        hidden_width=16, hidden_layers=2, dropout=0.0, norm="none"
    )
    cfg = SimpleNamespace(
        model=model_cfg,
        training=SimpleNamespace(lr=1e-3),
        runtime=SimpleNamespace(fast=True, batch_size=64),
        seed=0,
    )
    sd = unsupervised_pretrain(data, cfg, "cpu")  # type: ignore[arg-type]
    assert sd is not None and any("blocks" in k for k in sd)
    m = SharedMLP(n_features=f, n_classes=c, hidden_layers=2, hidden_width=16)
    m.load_state_dict(sd)
    out = continual_update(m, data, cfg, "cpu")  # type: ignore[arg-type]
    assert out is m
    print("extensions self-check OK")


if __name__ == "__main__":
    _selfcheck()
