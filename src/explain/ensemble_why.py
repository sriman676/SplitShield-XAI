"""Per-flow "why" for the deployed ENSEMBLE, not just its MLP member (owner 2026-10-15).

Purpose: the SOC engine explains an alert with gradient x input on the MLP member
only, although the alert score is a weighted average of the MLP and tree members
(``src/models/ensemble.py``). This module explains the score the analyst actually
sees:

  * MLP member: gradient x input of the predicted class's logit (same idea as
    ``engine._why``, re-implemented here so this module has no private imports);
  * LightGBM member: exact TreeSHAP, LightGBM's native
    ``booster.predict(X, pred_contrib=True)`` for the predicted class (raw-score /
    log-odds space);
  * Random Forest member (if deployed): TreeSHAP from the ``shap`` package when it
    is installed, otherwise left out (and reported as such).

Members' attributions live on different scales (logit vs log-odds vs probability),
so each member's vector is L1-normalised per flow before the ensemble weights are
applied: ``combined = sum_m w_m * a_m / ||a_m||_1 / sum_m w_m``. The combined value
is a signed SHARE of the explanation (positive = pushes toward the predicted
class), not a probability change. Raw per-member contributions are returned too.

Baselines differ: gradient x input is relative to the scaled origin (the training
mean after robust scaling); TreeSHAP is relative to the trees' expected output on
training data. Both mean "relative to typical training traffic".
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12


# --- member attributions --------------------------------------------------------


def mlp_attributions(mlp: Any, X: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Gradient x input of each row's class ``cls`` logit, shape (n, d)."""
    import torch

    x = torch.as_tensor(np.asarray(X, dtype=np.float32)).clone().requires_grad_(True)
    mlp.eval()
    logits = mlp(x)
    idx = torch.as_tensor(np.asarray(cls, dtype=np.int64))[:, None]
    logits.gather(1, idx).sum().backward()
    assert x.grad is not None
    return (x.grad * x).detach().numpy().astype(np.float64)


def _class_block(contrib: np.ndarray, n_feat: int, cls: np.ndarray) -> np.ndarray:
    """Pick each row's class block out of LightGBM's (n, (d+1)*K) pred_contrib."""
    k = contrib.shape[1] // (n_feat + 1)
    blocks = contrib.reshape(len(contrib), k, n_feat + 1)
    return blocks[np.arange(len(contrib)), cls, :n_feat]


def lgbm_attributions(model: Any, X: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """TreeSHAP of LightGBM for each row's class, shape (n, d).

    ``model`` is an ``LGBMClassifier`` (or a raw ``Booster``). ``cls`` indexes the
    deployment's class list; it is mapped onto the model's own ``classes_``. Binary
    LightGBM explains the positive class only, so class 0 gets the negated values.
    A class the model never saw gets zeros."""
    booster = getattr(model, "booster_", model)
    x = np.asarray(X, dtype=np.float64)
    d = x.shape[1]
    contrib = np.asarray(booster.predict(x, pred_contrib=True), dtype=np.float64)
    classes = list(np.asarray(getattr(model, "classes_", np.arange(2)), dtype=int))
    local = np.array([classes.index(c) if c in classes else -1 for c in cls])
    out = np.zeros((len(x), d))
    if contrib.shape[1] == d + 1:  # binary: one block for classes_[1]
        sign = np.where(local == 1, 1.0, np.where(local == 0, -1.0, 0.0))
        return contrib[:, :d] * sign[:, None]
    ok = local >= 0
    if ok.any():
        out[ok] = _class_block(contrib[ok], d, local[ok])
    return out


def rf_attributions(model: Any, X: np.ndarray, cls: np.ndarray) -> np.ndarray | None:
    """TreeSHAP of a Random Forest via ``shap`` (probability space); None if absent."""
    try:
        import shap
    except ImportError:
        return None
    x = np.asarray(X, dtype=np.float64)
    sv = shap.TreeExplainer(model).shap_values(x, check_additivity=False)
    arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    if arr.ndim == 2:  # single-output model
        arr = arr[:, :, None]
    classes = list(np.asarray(model.classes_, dtype=int))
    out = np.zeros(x.shape)
    for i, c in enumerate(cls):
        if c in classes:
            out[i] = arr[i, :, classes.index(c)]
    return out


# --- combination ----------------------------------------------------------------


def member_weights(det: Any) -> dict[str, float]:
    """Deployed member weights; MLP-only detectors give ``{"mlp": 1.0}``."""
    ens = getattr(det, "ensemble", None)
    return dict(ens.weights) if ens is not None else {"mlp": 1.0}


def predict_proba(det: Any, X: np.ndarray) -> np.ndarray:
    """Calibrated class probabilities of the deployed scorer (ensemble or MLP)."""
    import torch

    scorer = det.ensemble if getattr(det, "ensemble", None) is not None else det.model
    t = float(getattr(det, "temperature", 1.0) or 1.0)
    with torch.no_grad():
        logits = scorer(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        return torch.softmax(logits / t, 1).numpy().astype(np.float64)


def member_attributions(
    det: Any, X: np.ndarray, cls: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-member signed attributions (n, d) for every deployed member we can explain."""
    out: dict[str, np.ndarray] = {}
    trees = getattr(getattr(det, "ensemble", None), "trees", {}) or {}
    for name in member_weights(det):
        if name == "mlp":
            out[name] = mlp_attributions(det.model, X, cls)
        elif name == "lgbm" and name in trees:
            out[name] = lgbm_attributions(trees[name], X, cls)
        elif name == "rf" and name in trees:
            a = rf_attributions(trees[name], X, cls)
            if a is not None:
                out[name] = a
    return out


def combine(attrs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of per-flow L1-normalised member attributions (n, d)."""
    used = {k: float(weights.get(k, 0.0)) for k in attrs if weights.get(k, 0.0) > 0}
    if not used:
        raise ValueError("no explainable member with positive weight")
    total = sum(used.values())
    first = next(iter(attrs.values()))
    comb = np.zeros_like(first)
    for name, w in used.items():
        a = attrs[name]
        norm = np.abs(a).sum(axis=1, keepdims=True)
        comb += w * a / np.maximum(norm, EPS)
    return comb / total


# --- plain English --------------------------------------------------------------


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v:,.3g}"
    if a >= 100 or float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:.3g}"


def describe(name: str, value: float, typical: float | None) -> str:
    """'Flow Packets/s' is 40x higher than typical traffic (12,000 vs 300)."""
    if typical is None or not np.isfinite(typical):
        return f"'{name}' is {_fmt(value)}"
    if abs(typical) < EPS:
        if abs(value) < EPS:
            return f"'{name}' is 0, as in typical traffic"
        return f"'{name}' is {_fmt(value)} where typical traffic has 0"
    ratio = value / typical
    vs = f"({_fmt(value)} vs typical {_fmt(typical)})"
    if ratio >= 1.5:
        return f"'{name}' is {ratio:,.0f}x higher than typical traffic {vs}"
    if 0 < ratio <= 1 / 1.5:
        return f"'{name}' is {1 / ratio:,.0f}x lower than typical traffic {vs}"
    if ratio <= 0:
        return f"'{name}' is {_fmt(value)}, unlike typical traffic ({_fmt(typical)})"
    return f"'{name}' is close to typical traffic {vs}"


def _sentence(item: dict[str, Any], label: str) -> str:
    verb = "pushes it toward" if item["contribution"] > 0 else "argues against"
    share = abs(item["contribution"]) * 100
    return f"{describe(item['feature'], item['value'], item['typical'])}; this {verb} {label} ({share:.0f}% of the explanation)."


def _raw_values(det: Any, X: np.ndarray, X_raw: np.ndarray | None) -> np.ndarray:
    if X_raw is not None:
        return np.asarray(X_raw, dtype=np.float64)
    scaler = getattr(det, "scaler", None)
    if scaler is None:
        return np.asarray(X, dtype=np.float64)
    return np.asarray(scaler.inverse_transform(np.asarray(X, dtype=np.float64)))


def _typical(det: Any, n_feat: int) -> np.ndarray | None:
    stats = getattr(det, "input_stats", None) or {}
    med = stats.get("median")
    if med is None or len(med) != n_feat:
        return None
    return np.asarray(med, dtype=np.float64)


# --- public API -----------------------------------------------------------------


def explain_flows(
    det: Any,
    X_scaled: np.ndarray,
    rows_idx: list[int],
    k: int = 3,
    labels: list[str] | None = None,
    X_raw: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Top-``k`` signed feature contributions of the deployed ensemble per flow.

    Args:
        det: A loaded detector (``src/soc/model.Detector``-like: ``model``,
            ``ensemble``, ``feature_names``, ``label_names``, optional ``scaler``,
            ``input_stats``, ``temperature``).
        X_scaled: Model-space features for a batch of flows.
        rows_idx: Rows of ``X_scaled`` to explain.
        k: Features per flow.
        labels: Class to explain per row (default: the ensemble's prediction).
        X_raw: Raw-unit features (same rows as ``X_scaled``) for the sentences;
            default: ``det.scaler.inverse_transform``.

    Returns:
        One dict per row: ``row``, ``label``, ``probability``, ``members``,
        ``top`` (feature, contribution, per-member values, raw value, typical
        value) and ``text`` (numbered plain-English explanation).
    """
    if not rows_idx:
        return []
    X = np.asarray(X_scaled, dtype=np.float64)[rows_idx]
    names = list(det.label_names)
    proba = predict_proba(det, X)
    cls = proba.argmax(1)
    if labels is not None:
        cls = np.array(
            [
                names.index(lb) if lb in names else c
                for lb, c in zip(labels, cls, strict=True)
            ]
        )
    attrs = member_attributions(det, X, cls)
    comb = combine(attrs, member_weights(det))
    raw = _raw_values(det, X, None if X_raw is None else np.asarray(X_raw)[rows_idx])
    typical = _typical(det, X.shape[1])
    return [
        _flow_dict(
            det, j, r, names[cls[j]], proba[j, cls[j]], comb, attrs, raw, typical, k
        )
        for j, r in enumerate(rows_idx)
    ]


def _flow_dict(
    det: Any,
    j: int,
    row: int,
    label: str,
    prob: float,
    comb: np.ndarray,
    attrs: dict[str, np.ndarray],
    raw: np.ndarray,
    typical: np.ndarray | None,
    k: int,
) -> dict[str, Any]:
    feats = list(det.feature_names)
    order = np.argsort(-np.abs(comb[j]))[:k]
    top = [
        {
            "feature": feats[t],
            "contribution": float(comb[j, t]),
            "members": {m: float(a[j, t]) for m, a in attrs.items()},
            "value": float(raw[j, t]),
            "typical": None if typical is None else float(typical[t]),
        }
        for t in order
        if abs(comb[j, t]) > EPS
    ]
    text = "\n".join(f"{i}. {_sentence(it, label)}" for i, it in enumerate(top, 1))
    return {
        "row": int(row),
        "label": label,
        "probability": float(prob),
        "members": sorted(attrs),
        "top": top,
        "text": text,
    }
