"""SHAP explainer factory with the mandatory selection order (03_ARCHITECTURE §4).

``DeepExplainer`` is fragile across torch versions, so it is NOT the primary path:

    1. GradientExplainer   (primary)
    2. DeepExplainer       (fallback, failure logged not silenced)
    3. KernelExplainer     (final fallback) -- AND always run on a few instances
       as a cross-validation of whichever explainer was used.

A drift result built on a silently-wrong explainer is worthless, so the
KernelExplainer cross-validation Spearman is recorded in every metrics JSON that
contains attributions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from utils.logging_config import get_logger

_log = get_logger(__name__)


@dataclass
class ShapResult:
    """SHAP attributions for all classes plus explainer provenance."""

    attributions: np.ndarray  # shape (n_samples, n_features, n_classes)
    explainer: str
    kernel_crossval_spearman: float


def _proba_fn(
    model: torch.nn.Module, device: str
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a numpy predict-proba function for KernelExplainer / LIME."""

    @torch.no_grad()
    def f(x: np.ndarray) -> np.ndarray:
        model.eval()
        logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()

    return f


def _to_nfc(raw: Any, n: int, n_features: int, n_classes: int) -> np.ndarray:
    """Normalise SHAP output to shape (n_samples, n_features, n_classes)."""
    if isinstance(raw, list):  # list of (n, f) per class
        return np.stack(raw, axis=-1)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:  # single-class edge case
        return arr[:, :, None]
    raise ValueError(f"Unexpected SHAP output shape {arr.shape}")


def compute_shap(
    model: torch.nn.Module,
    X_explain: np.ndarray,
    background: np.ndarray,
    n_classes: int,
    device: str = "cpu",
    kernel_instances: int = 5,
) -> ShapResult:
    """Compute per-class SHAP attributions with cross-validated explainer choice.

    Args:
        model: A ``ChainedModel`` (or the shared MLP) over scaled features.
        X_explain: Instances to explain, shape (n, n_features), scaled.
        background: Background samples for the explainer, scaled.
        n_classes: Number of output classes (9).
        device: Torch device string.
        kernel_instances: Instances used for the KernelExplainer cross-validation.

    Returns:
        A ``ShapResult`` with attributions (n, f, C) and the cross-val Spearman.
    """
    import shap

    model.eval()
    n, n_features = X_explain.shape
    bg_t = torch.as_tensor(background, dtype=torch.float32, device=device)
    x_t = torch.as_tensor(X_explain, dtype=torch.float32, device=device)

    attributions: np.ndarray | None = None
    used = ""
    try:
        expl = shap.GradientExplainer(model, bg_t)
        attributions = _to_nfc(expl.shap_values(x_t), n, n_features, n_classes)
        used = "GradientExplainer"
    except Exception as exc:  # noqa: BLE001 - fall through, log, do not silence
        _log.warning("GradientExplainer failed (%s); trying DeepExplainer", exc)
        try:
            expl = shap.DeepExplainer(model, bg_t)
            attributions = _to_nfc(expl.shap_values(x_t), n, n_features, n_classes)
            used = "DeepExplainer"
        except Exception as exc2:  # noqa: BLE001
            _log.warning("DeepExplainer failed (%s); using KernelExplainer", exc2)

    # KernelExplainer on a few instances: final fallback AND always a cross-check.
    k = min(kernel_instances, n)
    kexpl = shap.KernelExplainer(_proba_fn(model, device), background)
    kernel_raw = kexpl.shap_values(X_explain[:k], nsamples=100, silent=True)
    kernel_attr = _to_nfc(kernel_raw, k, n_features, n_classes)

    if attributions is None:  # both gradient paths failed
        used = "KernelExplainer"
        attributions = _to_nfc(
            kexpl.shap_values(X_explain, nsamples=100, silent=True),
            n,
            n_features,
            n_classes,
        )

    crossval = _crossval_spearman(
        attributions[:k], kernel_attr, _proba_fn(model, device), X_explain[:k]
    )
    _log.info("SHAP via %s; KernelExplainer cross-val Spearman=%.3f", used, crossval)
    return ShapResult(
        attributions=attributions, explainer=used, kernel_crossval_spearman=crossval
    )


def _crossval_spearman(
    fast: np.ndarray,
    kernel: np.ndarray,
    proba: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
) -> float:
    """Mean Spearman between fast-explainer and KernelExplainer, predicted class."""
    pred = proba(X).argmax(axis=1)
    rhos: list[float] = []
    for i, c in enumerate(pred):
        c = min(int(c), fast.shape[2] - 1, kernel.shape[2] - 1)
        rho, _ = spearmanr(fast[i, :, c], kernel[i, :, c])
        if not np.isnan(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else 0.0
