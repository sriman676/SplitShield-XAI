"""SHAP attribution persistence and aggregation (03_ARCHITECTURE Stage 3).

Attributions are saved for ALL classes, not just the predicted one: Stage 6's
fixed-class drift needs the attribution vector for the true label held constant
(01_RESEARCH_DESIGN §4.4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.paths import ensure_within


def save_attributions(
    path: Path,
    attributions: np.ndarray,
    instance_ids: np.ndarray,
    class_names: tuple[str, ...],
    feature_names: list[str],
) -> None:
    """Persist per-class attributions and their instance ids to ``.npz``."""
    p = ensure_within(path)
    np.savez(
        p,
        attributions=attributions,  # (n, f, C)
        instance_ids=instance_ids,
        class_names=np.array(class_names),
        feature_names=np.array(feature_names),
    )


def global_mean_abs(
    attributions: np.ndarray, class_index: int | None = None
) -> np.ndarray:
    """Mean |SHAP| per feature, over samples (and classes unless one is given).

    Args:
        attributions: Array (n_samples, n_features, n_classes).
        class_index: If given, aggregate only that class; else average all.

    Returns:
        A per-feature importance vector of length n_features.
    """
    a = np.abs(attributions)
    if class_index is not None:
        return a[:, :, class_index].mean(axis=0)
    return a.mean(axis=(0, 2))


def topk_features(importance: np.ndarray, k: int = 10) -> set[int]:
    """Indices of the top-k features by importance magnitude."""
    return set(np.argsort(importance)[::-1][:k].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard overlap of two feature-index sets."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0
