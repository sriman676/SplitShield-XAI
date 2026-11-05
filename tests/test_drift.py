"""Explanation drift (§4.4) and the H2 one-sided paired test."""

from __future__ import annotations

import numpy as np

from src.eval.stats import wilcoxon_paired
from src.explain.drift import (
    fixed_class_matrix,
    per_instance_drift,
)


def test_fixed_class_matrix_selects_true_label_vector() -> None:
    # (n=2, f=3, C=2); pick class 0 for inst 0, class 1 for inst 1.
    attr = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    out = fixed_class_matrix(attr, np.array([0, 1]))
    assert out.shape == (2, 3)
    assert np.allclose(out[0], attr[0, :, 0])
    assert np.allclose(out[1], attr[1, :, 1])


def test_identical_attributions_have_zero_drift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(5, 8))
    d = per_instance_drift(ref, ref.copy(), k=3)
    assert np.allclose(d["spearman"], 0.0)
    assert np.allclose(d["cosine"], 0.0)
    assert np.allclose(d["jaccard"], 0.0)


def test_shuffled_attributions_drift_more_than_tiny_noise() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(6, 10))
    near = ref + rng.normal(scale=1e-3, size=ref.shape)
    shuffled = np.stack([rng.permutation(row) for row in ref])
    d_near = per_instance_drift(ref, near, k=3)
    d_shuf = per_instance_drift(ref, shuffled, k=3)
    assert d_shuf["spearman"].mean() > d_near["spearman"].mean()


def test_constant_vector_guard_no_nan() -> None:
    ref = np.zeros((3, 5))
    pert = np.ones((3, 5))
    d = per_instance_drift(ref, pert, k=2)
    for m in ("spearman", "jaccard", "cosine"):
        assert not np.isnan(d[m]).any()


def test_wilcoxon_greater_is_one_sided() -> None:
    adv = np.array([0.8, 0.7, 0.9, 0.75, 0.85])
    rnd = np.array([0.3, 0.35, 0.4, 0.25, 0.3])
    greater = wilcoxon_paired(adv, rnd, alternative="greater")
    two_sided = wilcoxon_paired(adv, rnd)
    assert greater["median_diff"] > 0
    # one-sided p is smaller (or equal) when the effect is in the tested direction
    assert greater["p"] <= two_sided["p"] + 1e-12
