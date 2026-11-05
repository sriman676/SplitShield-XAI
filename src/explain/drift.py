"""Explanation drift for H2 (01_RESEARCH_DESIGN §4.4, 03_ARCHITECTURE Stage 6).

The class-index rule is the whole point. An attack changes the predicted class,
so comparing "attribution for the predicted class" before and after compares
attributions for *different* classes and guarantees drift by construction. So:

- **Fixed-class drift (primary, the H2 quantity):** compare phi_c(x) vs phi_c(x')
  where ``c`` is the TRUE label, held fixed. ``fixed_class_matrix`` extracts it.
- **Predicted-class drift (secondary):** compare phi_{f(x)}(x) vs phi_{f(x')}(x').
  Report it, but label it confounded by the class change. Never the headline.

Drift = ``1 - similarity`` on three measures that can disagree (the disagreement
is informative): Spearman rank correlation over features (primary, robust to the
attribution-scale shifts a perturbation induces), top-k Jaccard (k=10), and raw
cosine similarity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.explain.shap_utils import jaccard, topk_features


def fixed_class_matrix(
    attributions: np.ndarray, class_indices: np.ndarray
) -> np.ndarray:
    """Select one attribution vector per instance at a held-fixed class index.

    Args:
        attributions: Array (n_samples, n_features, n_classes).
        class_indices: Length-n array; ``class_indices[i]`` is the class to keep
            for instance ``i`` (the true label for fixed-class drift).

    Returns:
        Array (n_samples, n_features) of the selected per-instance attributions.
    """
    n = attributions.shape[0]
    c = np.clip(np.asarray(class_indices, dtype=int), 0, attributions.shape[2] - 1)
    return attributions[np.arange(n), :, c]


def _spearman_sim(a: np.ndarray, b: np.ndarray) -> float:
    # spearmanr is undefined when a vector is constant; then rank order carries no
    # information, so treat identical vectors as perfectly similar, else unrelated.
    if a.std() == 0.0 or b.std() == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / (na * nb))


def per_instance_drift(
    ref: np.ndarray, pert: np.ndarray, k: int = 10
) -> dict[str, np.ndarray]:
    """Per-instance drift (``1 - similarity``) on all three measures (§4.4).

    Args:
        ref: Reference fixed-class attributions (n, f) for the clean instances.
        pert: Perturbed fixed-class attributions (n, f), paired with ``ref``.
        k: top-k for the Jaccard overlap (default 10 per §4.4).

    Returns:
        Dict with keys ``spearman``, ``jaccard``, ``cosine``; each a length-n array.
    """
    n = len(ref)
    kk = min(k, ref.shape[1])
    sp = np.empty(n)
    ja = np.empty(n)
    co = np.empty(n)
    for i in range(n):
        a, b = ref[i], pert[i]
        sp[i] = 1.0 - _spearman_sim(a, b)
        ja[i] = 1.0 - jaccard(
            topk_features(np.abs(a), kk), topk_features(np.abs(b), kk)
        )
        co[i] = 1.0 - _cosine_sim(a, b)
    return {"spearman": sp, "jaccard": ja, "cosine": co}


def drift_summary(drift: dict[str, np.ndarray]) -> dict[str, Any]:
    """Mean/std/n summary of a ``per_instance_drift`` result, for the metrics JSON."""
    return {
        m: {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
        for m, v in drift.items()
    }
