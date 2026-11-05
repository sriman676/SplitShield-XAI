"""Ensemble detector for DEPLOYMENT (owner decision 2026-10-14).

Tree models beat the neural net on some datasets (CIC-IoT-2023: RF 0.69 vs MLP
0.61 macro-F1), so deploy considers the MLP, LightGBM and a Random Forest alone and
as probability averages, and keeps whichever has the best VALIDATION macro-F1.

``EnsembleModel`` is an ``nn.Module`` whose forward returns log-probabilities of
the weighted average, so temperature fitting, alert thresholds and reports work on
it unchanged. Trees are not differentiable and log-probabilities have constant
energy, so SHAP indicators and the D3 energy detector keep using the MLP member
(``.mlp``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from utils.logging_config import get_logger

_log = get_logger(__name__)
EPS = 1e-7


class EnsembleModel(nn.Module):
    """Weighted probability average of the MLP and fitted sklearn-style models."""

    def __init__(
        self,
        mlp: Any,
        trees: dict[str, Any],
        weights: dict[str, float],
        log_bias: Any = None,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.n_classes = int(mlp.head.out_features)
        self.trees = trees
        self.weights = {k: float(v) for k, v in weights.items() if v > 0}
        # optional per-class log-prior shift chosen on VALIDATION (e.g. favour BENIGN
        # a little to hold the false-alarm rate after adding training data); None = off
        self.log_bias = (
            None
            if log_bias is None
            else torch.as_tensor(np.asarray(log_bias, dtype=np.float32))
        )

    def proba(self, x: torch.Tensor) -> torch.Tensor:
        total = sum(self.weights.values())
        p = torch.zeros(len(x), self.n_classes, device=x.device)
        if "mlp" in self.weights:
            p += self.weights["mlp"] * torch.softmax(self.mlp(x), 1)
        xn = x.detach().cpu().numpy()
        for name, w in self.weights.items():
            if name == "mlp":
                continue
            pt = _full_proba(self.trees[name], xn, p.shape[1])
            p += w * torch.as_tensor(pt, dtype=p.dtype, device=x.device)
        return p / total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.log(self.proba(x).clamp_min(EPS))
        if self.log_bias is not None:
            out = out + self.log_bias.to(out.device)
        return out


def _full_proba(model: Any, x: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba padded to all classes (a tree may not have seen every class)."""
    pr = model.predict_proba(x)
    out = np.zeros((len(x), n_classes))
    out[:, np.asarray(model.classes_, dtype=int)] = pr
    return out


WEIGHT_CAP = 50.0


def capped_weights(y: np.ndarray) -> np.ndarray:
    """Class-balanced sample weights, capped at WEIGHT_CAP x the median weight.

    Plain 'balanced' weights give a 7-row class (Heartbleed) a weight thousands of
    times larger than the majority; LightGBM's multiclass boosting then diverged on
    one NF-ToN-IoT seed (macro-F1 0.36 vs 0.91 on the others, owner 2026-10-15)."""
    from sklearn.utils.class_weight import compute_sample_weight

    w = compute_sample_weight("balanced", y)
    return np.minimum(w, WEIGHT_CAP * float(np.median(w)))


def fit_trees(data: Any, seed: int, fast: bool = False) -> dict[str, Any]:
    """LightGBM (capped balanced weights) and Random Forest (balanced_subsample)."""
    from sklearn.ensemble import RandomForestClassifier

    trees: dict[str, Any] = {}
    try:
        from lightgbm import LGBMClassifier

        gbm = LGBMClassifier(
            n_estimators=50 if fast else 600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        gbm.fit(data.X_train, data.y_train, sample_weight=capped_weights(data.y_train))
        trees["lgbm"] = gbm
    except ImportError:
        _log.warning("lightgbm not installed; ensemble without it")
    rf = RandomForestClassifier(
        n_estimators=20 if fast else 300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
        min_samples_leaf=1,
    )
    rf.fit(data.X_train, data.y_train)
    trees["rf"] = rf
    return trees


def candidates(trees: dict[str, Any]) -> list[dict[str, float]]:
    """Member weightings to compare: each alone, MLP + each tree, all together."""
    names = ["mlp", *trees]
    out = [{n: 1.0} for n in names]
    out += [{"mlp": 1.0, t: 1.0} for t in trees]
    if len(trees) > 1:
        out.append({n: 1.0 for n in names})
        out.append({t: 1.0 for t in trees})
    return out


def label(weights: dict[str, float]) -> str:
    return " + ".join(sorted(weights))


def choose(
    mlp: nn.Module,
    trees: dict[str, Any],
    data: Any,
    device: str,
    evaluate: Any,
) -> tuple[EnsembleModel, list[dict[str, Any]]]:
    """Best weighting on VALIDATION macro-F1 (``evaluate(model) -> float``)."""
    rows = []
    best: tuple[float, EnsembleModel] | None = None
    for w in candidates(trees):
        ens = EnsembleModel(mlp, trees, w).to(device).eval()
        v = float(evaluate(ens))
        rows.append({"members": label(w), "val_macro_f1": v})
        _log.info("ensemble %s: val macro-F1 %.4f", label(w), v)
        if best is None or v > best[0] + 1e-9:
            best = (v, ens)
    assert best is not None
    return best[1], rows
