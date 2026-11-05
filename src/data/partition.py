"""Client partitioning (04_DATA §5). Operates on the training split only.

Dirichlet label skew makes the non-IID knob an independent variable (H4) instead
of v1's hardcoded assignment. FLNET2023 native partitions are added in Task 9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

from src.data import labels as L
from utils.logging_config import get_logger
from utils.paths import ensure_within

_log = get_logger(__name__)


def partition_dirichlet(
    y_train: np.ndarray, n_clients: int, alpha: float, seed: int
) -> list[np.ndarray]:
    """Split training indices across clients by a Dir(alpha) label skew.

    Args:
        y_train: Integer family labels of the training split.
        n_clients: Number of clients.
        alpha: Dirichlet concentration; smaller => more skew.
        seed: Reproducible from ``(seed, alpha, n_clients)``.

    Returns:
        A list of index arrays (into ``y_train``), one per client.
    """
    rng = np.random.default_rng(seed)
    client_idx: list[list[int]] = [[] for _ in range(n_clients)]
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        for c, chunk in enumerate(np.split(cls_idx, cuts)):
            client_idx[c].extend(chunk.tolist())
    return [np.array(sorted(ci), dtype=np.int64) for ci in client_idx]


def _label_dist(y: np.ndarray) -> np.ndarray:
    """Normalised class distribution over the fixed FAMILY_ORDER."""
    counts = np.bincount(y, minlength=len(L.FAMILY_ORDER)).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def mean_pairwise_js(dists: list[np.ndarray]) -> float:
    """Mean pairwise Jensen-Shannon distance between client label distributions."""
    pairs = [
        jensenshannon(dists[i], dists[j], base=2)
        for i in range(len(dists))
        for j in range(i + 1, len(dists))
    ]
    vals = [float(p) for p in pairs if not np.isnan(p)]
    return float(np.mean(vals)) if vals else 0.0


def client_distribution_report(
    y_train: np.ndarray, partitions: list[np.ndarray], alpha: float
) -> dict[str, Any]:
    """Build the ``client_distributions.json`` payload (04_DATA §5.3)."""
    dists = [_label_dist(y_train[p]) for p in partitions]
    return {
        "alpha": alpha,
        "n_clients": len(partitions),
        "realised_mean_pairwise_js": mean_pairwise_js(dists),
        "clients": [
            {
                "total": int(len(p)),
                "class_counts": {
                    L.FAMILY_ORDER[i]: int(n)
                    for i, n in zip(
                        *np.unique(y_train[p], return_counts=True), strict=False
                    )
                },
            }
            for p in partitions
        ],
    }


def write_client_distributions(path: Path, report: dict[str, Any]) -> None:
    """Write the client-distribution report as JSON inside the project tree."""
    p = ensure_within(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _log.info(
        "Wrote client distributions (mean pairwise JS=%.3f) to %s",
        report["realised_mean_pairwise_js"],
        p,
    )
