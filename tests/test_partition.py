"""Dirichlet partitioning: reproducible, lossless, and skew is measurable."""

from __future__ import annotations

import numpy as np

from src.data.partition import (
    client_distribution_report,
    partition_dirichlet,
)


def _y() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 9, size=400)


def test_partition_is_reproducible() -> None:
    a = partition_dirichlet(_y(), n_clients=4, alpha=0.5, seed=1)
    b = partition_dirichlet(_y(), n_clients=4, alpha=0.5, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=False))


def test_partition_is_lossless_and_disjoint() -> None:
    y = _y()
    parts = partition_dirichlet(y, n_clients=4, alpha=1.0, seed=2)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(y)
    assert len(set(allidx.tolist())) == len(y)  # no index assigned twice


def test_more_skew_raises_js_divergence() -> None:
    y = _y()
    iid = client_distribution_report(y, partition_dirichlet(y, 4, 10.0, 3), 10.0)
    skew = client_distribution_report(y, partition_dirichlet(y, 4, 0.1, 3), 0.1)
    assert skew["realised_mean_pairwise_js"] > iid["realised_mean_pairwise_js"]
