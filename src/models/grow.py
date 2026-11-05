"""Dynamic neuron growth for the DEPLOYED detector (owner decision 2026-10-14).

Research runs keep the registered fixed-size SharedMLP. The deployed detector may
grow itself when more capacity helps:

  1. widen  - every hidden layer gets ``WIDEN`` x more neurons (Net2WiderNet,
     Chen et al. 2016): new neurons copy existing ones and the next layer's
     incoming weights are split between copies, so the grown network computes the
     same function before it trains on (nothing learned is lost);
  2. deepen - when widening stops helping, an identity block is inserted before
     the head (Net2DeeperNet: identity Linear + ReLU on non-negative inputs);
  3. keep a step only if VALIDATION macro-F1 improves by more than ``MIN_GAIN``;
  4. prune back - neurons with the lowest importance (mean |activation| x outgoing
     weight norm on validation) are removed in uniform steps while validation stays
     within ``PRUNE_TOL`` of the best, so the final model is no larger than needed;
  5. prune weak links - the smallest-magnitude individual connections (global
     magnitude pruning, Han et al. 2015) are zeroed in growing fractions, each level
     briefly fine-tuned with the zeros held fixed, while validation stays within
     ``PRUNE_TOL`` (removes noisy connections; the sparsity is reported).

Widths stay uniform across layers so the result is still a plain SharedMLP (the
SOC Detector, SHAP and the energy detector load it unchanged). Exactly
function-preserving with ``norm="none"`` (the default); with layer/batch norm the
copy is approximate and the continued training repairs it.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.mlp import SharedMLP
from utils.logging_config import get_logger

_log = get_logger(__name__)

WIDEN = 1.5
MAX_WIDTH = 1024
MAX_LAYERS = 8
MIN_GAIN = 0.002  # val macro-F1 a growth step must add to be kept
PRUNE_TOL = 0.002  # val macro-F1 a prune step may cost
PRUNE_STEP = 0.25  # fraction of neurons removed per prune step
LINK_LEVELS = (0.2, 0.4, 0.6, 0.75, 0.85, 0.9)  # share of connections zeroed
LINK_EPOCHS = 2  # masked fine-tune epochs per level
NOISE = 1e-3  # breaks the symmetry between a neuron and its copy


def _linear(block: Any) -> nn.Linear:
    lin = block[0]
    assert isinstance(lin, nn.Linear)
    return lin


def _norm_layer(block: Any) -> nn.Module | None:
    m = block[1]
    return m if isinstance(m, nn.LayerNorm | nn.BatchNorm1d) else None


def _resize_norm(norm: nn.Module, idx: torch.Tensor) -> None:
    """Re-index a norm layer's per-neuron parameters/statistics to ``idx``."""
    for name in ("weight", "bias"):
        p = getattr(norm, name, None)
        if p is not None:
            setattr(norm, name, nn.Parameter(p.data[idx].clone()))
    for name in ("running_mean", "running_var"):
        b = getattr(norm, name, None)
        if b is not None:
            setattr(norm, name, b[idx].clone())
    if isinstance(norm, nn.LayerNorm):
        norm.normalized_shape = (len(idx),)
    elif isinstance(norm, nn.BatchNorm1d):
        norm.num_features = len(idx)


def _reindex(
    model: SharedMLP, keep: list[torch.Tensor], split: list[torch.Tensor] | None
) -> SharedMLP:
    """New model whose layer i has the neurons ``keep[i]`` (indices into the old
    layer, repeats allowed). ``split[i]`` divides the NEXT layer's incoming weight
    of each kept neuron (replica count when widening; ones when pruning)."""
    m: Any = copy.deepcopy(model)
    block: Any
    for i, block in enumerate(m.blocks):
        lin = _linear(block)
        idx = keep[i]
        w = lin.weight.data[idx].clone()
        b = lin.bias.data[idx].clone()
        if i > 0:  # incoming side follows the previous layer's selection
            prev, div = keep[i - 1], (split[i - 1] if split else None)
            w = w[:, prev] / (div if div is not None else 1.0)
        if split is not None:  # a copied neuron: tiny noise breaks the symmetry
            fresh = torch.zeros(len(idx), dtype=torch.bool)
            seen: set[int] = set()
            for j, k in enumerate(idx.tolist()):
                fresh[j] = k in seen
                seen.add(k)
            fresh = fresh.to(w.device)
            w[fresh] += NOISE * torch.randn_like(w[fresh])
        new = nn.Linear(w.shape[1], w.shape[0]).to(w.device)
        new.weight.data, new.bias.data = w, b
        block[0] = new
        norm = _norm_layer(block)
        if norm is not None:
            _resize_norm(norm, idx)
    head = m.head
    last, div = keep[-1], (split[-1] if split else None)
    hw = head.weight.data[:, last] / (div if div is not None else 1.0)
    new_head = nn.Linear(hw.shape[1], hw.shape[0]).to(hw.device)
    new_head.weight.data, new_head.bias.data = hw, head.bias.data.clone()
    m.head = new_head
    return m


def widen(model: SharedMLP, new_width: int, seed: int = 0) -> SharedMLP:
    """Net2WiderNet: every hidden layer -> ``new_width`` neurons, same function."""
    g = torch.Generator().manual_seed(seed)
    keep, split = [], []
    for block in model.blocks:
        w = _linear(block).out_features
        extra = torch.randint(0, w, (new_width - w,), generator=g)
        idx = torch.cat([torch.arange(w), extra])
        counts = torch.bincount(idx, minlength=w).float()
        keep.append(idx)
        split.append(counts[idx].unsqueeze(0).to(_linear(block).weight.device))
    return _reindex(model, keep, split)


def deepen(model: SharedMLP, dropout: float) -> SharedMLP:
    """Net2DeeperNet: insert an identity block before the head, same function
    (inputs to it are post-ReLU, i.e. non-negative, so ReLU(I x) = x)."""
    m: Any = copy.deepcopy(model)
    last: Any = m.blocks[-1]
    width = _linear(last).out_features
    block: Any = copy.deepcopy(last)
    lin = nn.Linear(width, width).to(_linear(last).weight.device)
    nn.init.eye_(lin.weight)
    nn.init.zeros_(lin.bias)
    block[0] = lin
    norm = _norm_layer(block)
    if isinstance(norm, nn.LayerNorm):
        nn.init.ones_(norm.weight)
        nn.init.zeros_(norm.bias)
    if isinstance(block[-1], nn.Dropout):
        block[-1].p = dropout
    m.blocks.append(block)
    return m


@torch.no_grad()
def neuron_importance(model: SharedMLP, X: np.ndarray, device: str) -> list[np.ndarray]:
    """Per layer: mean |activation| on ``X`` x L2 norm of the outgoing weights."""
    model.eval()
    x = torch.as_tensor(np.asarray(X[:20000]), dtype=torch.float32, device=device)
    acts = []
    for block in model.blocks:
        x = block(x)
        acts.append(x.abs().mean(0))
    outs = [_linear(b).weight for b in list(model.blocks)[1:]] + [model.head.weight]
    return [(a * w.norm(dim=0)).cpu().numpy() for a, w in zip(acts, outs, strict=True)]


def prune(model: SharedMLP, width: int, X: np.ndarray, device: str) -> SharedMLP:
    """Keep the ``width`` most important neurons of every hidden layer."""
    imp = neuron_importance(model, X, device)
    keep = [
        torch.as_tensor(np.sort(np.argsort(-s)[:width]), device=device) for s in imp
    ]
    return _reindex(model, keep, None)


def _weights(model: Any) -> list[torch.Tensor]:
    return [_linear(b).weight for b in model.blocks] + [model.head.weight]


def link_masks(model: Any, sparsity: float) -> list[torch.Tensor]:
    """Keep-masks zeroing the globally smallest ``sparsity`` share of connections."""
    ws = _weights(model)
    flat = torch.cat([w.detach().abs().flatten() for w in ws])
    k = int(sparsity * flat.numel())
    if k == 0:
        return [torch.ones_like(w, dtype=torch.bool) for w in ws]
    thr = flat.kthvalue(k).values
    return [w.detach().abs() > thr for w in ws]


def _apply(model: Any, masks: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for w, m in zip(_weights(model), masks, strict=True):
            w.mul_(m)


def sparsity_of(model: Any) -> float:
    ws = _weights(model)
    return float(sum(int((w == 0).sum()) for w in ws) / sum(w.numel() for w in ws))


def masked_finetune(
    model: Any,
    masks: list[torch.Tensor],
    data: Any,
    device: str,
    epochs: int,
    lr: float,
) -> Any:
    """Fine-tune with pruned connections held at zero (mask re-applied each step)."""
    m = copy.deepcopy(model).to(device)
    _apply(m, masks)
    x = torch.as_tensor(np.asarray(data.X_train), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(data.y_train), dtype=torch.long)
    cw = getattr(data, "class_weights", None)
    loss_fn = nn.CrossEntropyLoss(
        weight=None
        if cw is None
        else torch.as_tensor(cw, dtype=torch.float32, device=device)
    )
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    m.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=g).split(1024):
            opt.zero_grad()
            loss_fn(m(x[idx].to(device)), y[idx].to(device)).backward()
            opt.step()
            _apply(m, masks)
    return m.eval()


def prune_links(
    model: Any, data: Any, device: str, evaluate: Any, best_val: float, lr: float
) -> tuple[Any, float, list[dict[str, Any]]]:
    """Step 5: zero ever more of the weakest connections while validation holds."""
    log: list[dict[str, Any]] = []
    for level in LINK_LEVELS:
        cand = masked_finetune(
            model, link_masks(model, level), data, device, LINK_EPOCHS, lr
        )
        v = evaluate(cand)
        ok = v >= best_val - PRUNE_TOL
        log.append({"step": "prune_links", "sparsity": level, "val_before": best_val,
                    "val_after": v, "kept": ok})  # fmt: skip
        _log.info("grow: prune links %.0f%% -> val %.4f (%s)", 100 * level, v,
                  "kept" if ok else "stop")  # fmt: skip
        if not ok:
            break
        model, best_val = cand, max(best_val, v)
    return model, best_val, log


def grow(
    data: Any,
    cfg: Any,
    model: Any,
    device: str,
    n_classes: int,
    train: Any,
    evaluate: Any,
    fit: dict[str, Any] | None = None,
) -> tuple[Any, SharedMLP, list[dict[str, Any]]]:
    """Grow -> keep only validated gains -> prune back. ``train(data, cfg, device,
    init_state)`` continues training (returns model, _, metrics); ``evaluate(model)``
    is validation macro-F1. Returns the final (cfg, model, log)."""
    log: list[dict[str, Any]] = []
    best_val = evaluate(model)
    start_width = cfg.model.hidden_width

    def attempt(kind: str, cand_cfg: Any, cand: SharedMLP) -> bool:
        nonlocal cfg, model, best_val, fit
        m2, _, met = train(data, cand_cfg, device, init_state=cand.state_dict())
        v = evaluate(m2)
        kept = v > best_val + MIN_GAIN
        log.append(
            {
                "step": kind,
                "layers": cand_cfg.model.hidden_layers,
                "width": cand_cfg.model.hidden_width,
                "val_before": best_val,
                "val_after": v,
                "fit_after": met.get("fit", {}).get("verdict"),
                "kept": kept,
            }  # fmt: skip
        )
        _log.info(
            "grow: %s -> %dx%d val %.4f -> %.4f (%s)",
            kind, cand_cfg.model.hidden_layers, cand_cfg.model.hidden_width,
            best_val, v, "kept" if kept else "reverted",
        )  # fmt: skip
        if kept:
            cfg, model, best_val, fit = cand_cfg, m2, v, met.get("fit")
        return kept

    # 1-3: grow while it pays; widen first, deepen when widening stops helping
    while True:
        w = cfg.model.hidden_width
        if w < MAX_WIDTH:
            nw = min(MAX_WIDTH, int(round(w * WIDEN)))
            c = replace(cfg, model=replace(cfg.model, hidden_width=nw))
            if attempt("widen", c, widen(model, nw, seed=len(log))):
                continue
        if cfg.model.hidden_layers < MAX_LAYERS:
            c = replace(
                cfg, model=replace(cfg.model, hidden_layers=cfg.model.hidden_layers + 1)
            )
            if attempt("deepen", c, deepen(model, cfg.model.dropout)):
                continue
        break

    # 4: prune back to the smallest uniform width that keeps validation
    grown_width = cfg.model.hidden_width
    while cfg.model.hidden_width > 8:
        nw = max(8, int(cfg.model.hidden_width * (1 - PRUNE_STEP)))
        cand = prune(model, nw, data.X_val, device)
        c = replace(cfg, model=replace(cfg.model, hidden_width=nw))
        tune = replace(
            c,
            training=replace(
                c.training,
                max_epochs=max(3, c.training.max_epochs // 5),
                lr=c.training.lr * 0.5,
            ),
        )
        m2, _, _ = train(data, tune, device, init_state=cand.state_dict())
        v = evaluate(m2)
        ok = v >= best_val - PRUNE_TOL
        log.append(
            {
                "step": "prune",
                "layers": c.model.hidden_layers,
                "width": nw,
                "val_before": best_val,
                "val_after": v,
                "kept": ok,
            }  # fmt: skip
        )
        _log.info("grow: prune -> width %d val %.4f (%s)", nw, v,
                  "kept" if ok else "stop")  # fmt: skip
        if not ok:
            break
        cfg, model = c, m2
        best_val = max(best_val, v)
    # 5: prune weak links (individual connections)
    model, best_val, links = prune_links(
        model, data, device, evaluate, best_val, cfg.training.lr * 0.2
    )
    log += links
    _log.info(
        "grow: width %d -> grown %d -> final %dx%d (val %.4f)",
        start_width, grown_width, cfg.model.hidden_layers, cfg.model.hidden_width,
        best_val,
    )  # fmt: skip
    return cfg, model, log
