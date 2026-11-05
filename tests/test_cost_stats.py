"""Client cost (§4.6) and the H1 trend statistics (§5) for the cut-depth headline."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.cost import client_cost
from src.eval.stats import benjamini_hochberg, spearman_trend, wilcoxon_paired


def test_client_cost_rises_with_depth() -> None:
    c1 = client_cost(n_features=20, hidden_width=128, cut_depth=1)
    c2 = client_cost(n_features=20, hidden_width=128, cut_depth=2)
    c3 = client_cost(n_features=20, hidden_width=128, cut_depth=3)
    assert (
        c1["client_forward_flops"]
        < c2["client_forward_flops"]
        < c3["client_forward_flops"]
    )
    assert c1["client_params"] < c2["client_params"] < c3["client_params"]
    # uniform width => cut-activation size is constant across depth (reported, not hidden)
    assert (
        c1["cut_activation_bytes_per_sample"] == c3["cut_activation_bytes_per_sample"]
    )


def test_client_cost_flops_formula() -> None:
    c = client_cost(n_features=10, hidden_width=8, cut_depth=1)
    assert c["client_forward_flops"] == 2 * 10 * 8
    assert c["client_params"] == 10 * 8 + 8


def test_spearman_trend_detects_monotone_decrease() -> None:
    # leakage decreasing as depth rises, across seeds (H1 prediction P1)
    levels = np.array([1, 1, 2, 2, 3, 3])
    values = np.array([0.9, 0.85, 0.6, 0.55, 0.3, 0.2])
    res = spearman_trend(levels, values)
    assert res["rho"] < 0
    assert res["p"] < 0.05
    assert res["n"] == 6


def test_spearman_trend_flat_is_not_significant() -> None:
    levels = np.array([1, 1, 2, 2, 3, 3])
    values = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    res = spearman_trend(levels, values)
    assert res["p"] == pytest.approx(1.0)


def test_wilcoxon_paired_reports_effect_and_direction() -> None:
    adv = np.array([0.8, 0.7, 0.9, 0.75, 0.85])
    rnd = np.array([0.3, 0.35, 0.4, 0.25, 0.3])
    res = wilcoxon_paired(adv, rnd)
    assert res["median_diff"] > 0
    assert res["p"] < 0.1


def test_benjamini_hochberg_order_preserved_and_bounded() -> None:
    raw = [0.01, 0.04, 0.03, 0.5]
    adj = benjamini_hochberg(raw)
    assert len(adj) == 4
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert all(a >= r for a, r in zip(adj, raw, strict=True))  # adjusted >= raw
