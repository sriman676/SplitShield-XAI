"""Statistics for headline claims (01_RESEARCH_DESIGN §5).

Shared so no stage rolls its own test. With n=5 seeds these tests are weak; treat
p-values as descriptive, not confirmatory (§5, 11_LIMITATIONS §1). Effect sizes
are reported alongside p, never p alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr, wilcoxon


def spearman_trend(levels: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Spearman rank correlation of ``values`` against ordinal ``levels`` (H1).

    Pass one (level, value) pair per observation — e.g. every ``(d, leakage)``
    across depths and seeds. ``rho`` is the effect size; ``p`` its significance.
    """
    levels = np.asarray(levels, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(np.unique(values)) < 2 or len(np.unique(levels)) < 2:
        return {"rho": 0.0, "p": 1.0, "n": int(len(values))}
    rho, p = spearmanr(levels, values)
    return {"rho": float(rho), "p": float(p), "n": int(len(values))}


def wilcoxon_paired(
    a: np.ndarray, b: np.ndarray, alternative: str = "two-sided"
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank (H2/H3), with a median-difference effect size.

    Tests ``a - b``. ``alternative`` defaults to two-sided; H2's falsification is
    directional ("paired difference not significantly *greater* than zero",
    §1 H2), so Stage 6 passes ``alternative="greater"`` to match the
    pre-registered criterion exactly. Returns median difference so the direction
    and magnitude are reported, not only the p-value.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p": 1.0, "median_diff": 0.0, "n": int(len(diff))}
    stat, p = wilcoxon(a, b, alternative=alternative)
    return {
        "statistic": float(stat),
        "p": float(p),
        "median_diff": float(np.median(diff)),
        "n": int(len(diff)),
    }


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (§5 multiple comparisons).

    Preserves input order. Report both raw and adjusted p within each hypothesis
    family.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    adj_sorted = p[order] * n / ranks
    # enforce monotonicity (step-up), then clip to 1
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj = np.empty(n, dtype=np.float64)
    adj[order] = np.clip(adj_sorted, 0.0, 1.0)
    return [float(x) for x in adj]
