"""Model compression so a trained detector runs on a gaming/low-end laptop.

Three levers, all reproducible:
  * width distillation  -> fewer hidden nodes (a smaller SharedMLP student trained
    on the teacher's soft logits + the true labels);
  * dynamic int8 quantisation -> ~4x smaller Linear weights, faster CPU inference;
  * measurement -> state-dict bytes + single-batch latency, so promotion can be
    gated on size/speed, not just accuracy.

Width distillation is the "delete unnecessary nodes" step: the student keeps the
SharedMLP shape (so splittable.py / SHAP still work) but with a narrower width the
search picked. Nothing here mutates the teacher.
"""

from __future__ import annotations

import io
import time

import numpy as np
import torch
from torch import nn

from config.base import Config
from src.data.preprocess import ProcessedData
from src.models.mlp import SharedMLP, param_count
from src.training.common import EarlyStopper, eval_macro_f1, make_loader


def state_dict_bytes(model: nn.Module) -> int:
    """Serialised size of the model weights in bytes (what ships to the laptop)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


@torch.no_grad()
def latency_ms(model: nn.Module, x: np.ndarray, device: str, repeats: int = 3) -> float:
    """Mean forward latency for one batch, in milliseconds (deployment proxy)."""
    model.eval()
    xb = torch.as_tensor(x, dtype=torch.float32, device=device)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(xb)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def distill_to_width(
    teacher: nn.Module,
    data: ProcessedData,
    cfg: Config,
    device: str,
    hidden_layers: int,
    hidden_width: int,
    dropout: float,
    temperature: float = 3.0,
    alpha: float = 0.5,
) -> tuple[nn.Module, float]:
    """Train a narrower student to match the teacher; return (student, val_f1).

    Loss = alpha * KL(student||teacher soft) + (1-alpha) * CE(student, true label).
    Early-stops on validation macro-F1 exactly like the base trainer, so the
    student is a drop-in replacement for the teacher at a fraction of the width.
    """
    n_classes = len(data.label_names)
    student = SharedMLP(
        n_features=len(data.feature_names),
        n_classes=n_classes,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        dropout=dropout,
        norm=cfg.model.norm,
    ).to(device)

    weight = (
        torch.as_tensor(data.class_weights, dtype=torch.float32, device=device)
        if data.class_weights is not None
        else None
    )
    ce = nn.CrossEntropyLoss(weight=weight)
    kl = nn.KLDivLoss(reduction="batchmean")
    opt = torch.optim.AdamW(
        student.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    loader = make_loader(
        data.X_train, data.y_train, cfg.runtime.batch_size, cfg.seed, cfg.runtime
    )
    stopper = EarlyStopper(cfg.training.patience)

    teacher.eval()
    t = temperature
    for _ in range(cfg.training.max_epochs):
        student.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                soft = torch.softmax(teacher(xb) / t, dim=1)
            opt.zero_grad()
            logits = student(xb)
            loss = alpha * (t * t) * kl(torch.log_softmax(logits / t, dim=1), soft) + (
                1 - alpha
            ) * ce(logits, yb)
            loss.backward()
            opt.step()
        vf1 = eval_macro_f1(student, data.X_val, data.y_val, n_classes, device)
        if stopper.step(vf1, student):
            break
    stopper.restore(student)
    return student, stopper.best


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Int8-quantise Linear layers for CPU inference (no accuracy retraining).

    Returns a CPU model; quantised weights cut size ~4x. Kept separate from the
    promotion metric so the un-quantised student is what SHAP/attacks still use.
    """
    return torch.ao.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear}, dtype=torch.qint8
    )


def teacher_params(model: nn.Module) -> int:
    """Convenience re-export so the loop needn't import mlp for one call."""
    return param_count(model)
