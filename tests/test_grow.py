"""Dynamic neuron growth (src/models/grow.py): function-preserving widen/deepen,
importance pruning, and the keep-only-if-validation-improves loop."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.models import grow as g
from src.models.mlp import SharedMLP


def _model(width: int = 16, layers: int = 3, norm: str = "none") -> SharedMLP:
    torch.manual_seed(0)
    return SharedMLP(10, 4, layers, width, 0.2, norm).eval()


def test_widen_and_deepen_preserve_function() -> None:
    m = _model()
    x = torch.randn(64, 10)
    y = m(x)
    w = g.widen(m, 24).eval()
    blocks: Any = w.blocks
    assert all(b[0].out_features == 24 for b in blocks)
    torch.testing.assert_close(w(x), y, atol=5e-2, rtol=0)  # only symmetry noise
    d = g.deepen(m, 0.2).eval()
    assert len(d.blocks) == 4
    torch.testing.assert_close(d(x), y)
    # still loadable as a plain SharedMLP (what the SOC Detector does)
    SharedMLP(10, 4, 4, 16, 0.2).load_state_dict(d.state_dict())
    SharedMLP(10, 4, 3, 24, 0.2).load_state_dict(w.state_dict())


def test_prune_keeps_important_neurons() -> None:
    m = _model(width=16)
    with torch.no_grad():  # neuron 5 of every layer is dead -> least important
        b: Any
        for b in m.blocks:
            b[0].weight[5] = 0
            b[0].bias[5] = -1
    x = np.random.default_rng(0).normal(size=(200, 10)).astype("float32")
    imp = g.neuron_importance(m, x, "cpu")
    assert all(s[5] == s.min() for s in imp)
    p = g.prune(m, 15, x, "cpu").eval()
    t = torch.as_tensor(x)
    torch.testing.assert_close(p(t), m(t), atol=1e-5, rtol=0)  # dead unit removed
    SharedMLP(10, 4, 3, 15, 0.2).load_state_dict(p.state_dict())


def test_grow_keeps_only_validated_steps() -> None:
    from dataclasses import dataclass  # grow() uses dataclasses.replace on cfg

    @dataclass(frozen=True)
    class M:
        hidden_width: int
        hidden_layers: int
        dropout: float

    @dataclass(frozen=True)
    class T:
        max_epochs: int
        lr: float

    @dataclass(frozen=True)
    class C:
        model: M
        training: T

    cfg = C(M(16, 3, 0.2), T(10, 1e-3))
    x = np.random.default_rng(0).normal(size=(100, 10)).astype("float32")
    y = np.random.default_rng(1).integers(0, 4, 100)
    data = SimpleNamespace(X_val=x, X_train=x, y_train=y, class_weights=None)

    def train(data, c, device, init_state=None):  # noqa: ANN001, ANN202
        m = SharedMLP(10, 4, c.model.hidden_layers, c.model.hidden_width, 0.2)
        m.load_state_dict(init_state)
        return m, None, {"fit": {"verdict": "good fit"}}

    # validation rewards width up to 36 neurons, depth never, then flat
    def evaluate(m):  # noqa: ANN001, ANN202
        blocks: Any = m.blocks
        w, depth = blocks[0][0].out_features, len(blocks)
        return min(w, 36) / 100 - 0.01 * (depth - 3)

    c, m, log = g.grow(data, cfg, _model(), "cpu", 4, train, evaluate)
    steps = [(e["step"], e.get("width"), e["kept"]) for e in log]
    assert ("widen", 24, True) in steps and ("widen", 36, True) in steps
    assert ("widen", 54, False) in steps and ("deepen", 36, False) in steps
    prunes = [e for e in log if e["step"] == "prune"]
    assert prunes and not prunes[-1]["kept"]  # neuron pruning stops at a loss
    assert log[-1]["step"] == "prune_links"  # then weak links are tried
    assert c.model.hidden_width == 36 and c.model.hidden_layers == 3
    assert replace(c.model) == c.model


def test_weak_links_pruned_and_held_at_zero() -> None:
    m = _model(width=16)
    masks = g.link_masks(m, 0.5)
    kept = sum(int(k.sum()) for k in masks)
    total = sum(k.numel() for k in masks)
    assert abs(kept / total - 0.5) < 0.01
    rng = np.random.default_rng(0)
    data = SimpleNamespace(
        X_train=rng.normal(size=(300, 10)).astype("float32"),
        y_train=rng.integers(0, 4, 300),
        class_weights=None,
    )
    tuned = g.masked_finetune(m, masks, data, "cpu", epochs=2, lr=1e-2)
    assert g.sparsity_of(tuned) >= 0.49  # zeros survived training

    # prune_links stops at the first level that costs validation
    calls = iter([0.50, 0.50, 0.40])
    out, best, log = g.prune_links(m, data, "cpu", lambda _m: next(calls), 0.5, 1e-3)
    assert [e["kept"] for e in log] == [True, True, False]
    assert g.sparsity_of(out) >= 0.39
