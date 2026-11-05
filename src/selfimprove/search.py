"""Reproducible architecture search space for the self-improvement loop.

A small seeded random search over width/depth/dropout is enough to find a compact
student that holds macro-F1 — NAS proper is overkill for a 4-layer MLP. Widths and
depths are only ever <= the teacher's, since the goal is to shrink.
"""

from __future__ import annotations

import numpy as np

# ponytail: fixed candidate grids, not a learned controller. A 4-layer tabular MLP
# has a tiny useful search space; add a real NAS controller only if this plateaus.
_WIDTHS = (16, 24, 32, 48, 64, 96, 128)
_LAYERS = (2, 3, 4)
_DROPOUTS = (0.1, 0.2, 0.3)


def sample_archs(
    n: int, seed: int, max_width: int = 128, max_layers: int = 4
) -> list[dict[str, float | int]]:
    """Return ``n`` distinct (width, layers, dropout) candidates, smallest first.

    Bounded by the teacher's width/depth so every candidate is a compression.
    Deterministic in ``seed``. Sorted by parameter proxy (width*layers) so the loop
    tries the cheapest models first and can stop early once one is good enough.
    """
    rng = np.random.default_rng(seed)
    widths = [w for w in _WIDTHS if w <= max_width] or [max_width]
    layers = [ln for ln in _LAYERS if ln <= max_layers] or [max_layers]
    seen: set[tuple[int, int, float]] = set()
    out: list[dict[str, float | int]] = []
    # exhaustive space is small; sample without replacement up to its size
    budget = min(n, len(widths) * len(layers) * len(_DROPOUTS))
    while len(out) < budget:
        cand = (
            int(rng.choice(widths)),
            int(rng.choice(layers)),
            float(rng.choice(_DROPOUTS)),
        )
        if cand in seen:
            continue
        seen.add(cand)
        out.append(
            {"hidden_width": cand[0], "hidden_layers": cand[1], "dropout": cand[2]}
        )
    out.sort(key=lambda a: int(a["hidden_width"]) * int(a["hidden_layers"]))
    return out
