"""Live traffic drift vs the training data (owner decision 2026-10-14).

The SOC keeps a rolling sample of scored flows (``flow_sample.jsonl``). Each
feature is binned by the training deciles stored at deploy time; the Population
Stability Index (PSI) compares the live share per bin with the training 10 %:

    PSI = sum_bins (live - train) * ln(live / train)

PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 a large shift (common credit-risk /
MLOps rule of thumb). Traffic counts as drifted when more than ``SHARE`` of the
features shift strongly - the autopilot then adapts or asks for reviews.
"""

from __future__ import annotations

from typing import Any

import numpy as np

STRONG = 0.25
SHARE = 0.2
MIN_ROWS = 200
EPS = 1e-4


def psi_per_feature(x: np.ndarray, deciles: list[list[float]]) -> np.ndarray:
    """PSI of each column of ``x`` against its training deciles."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape[1])
    for j, edges in enumerate(deciles):
        e = np.unique(np.asarray(edges, dtype=np.float64))
        bins = np.searchsorted(e, x[:, j], side="right")
        live = np.bincount(bins, minlength=len(e) + 1) / max(1, len(x))
        # training share per bin: decile gaps (merged when deciles tie)
        q = np.concatenate([[0.0], np.searchsorted(edges, e, side="right") / 10, [1.0]])
        train = np.diff(q)
        live, train = np.clip(live, EPS, None), np.clip(train, EPS, None)
        out[j] = float(np.sum((live - train) * np.log(live / train)))
    return out


def report(
    x: np.ndarray, names: list[str], deciles: list[list[float]]
) -> dict[str, Any]:
    if len(x) < MIN_ROWS:
        return {"rows": len(x), "status": "not_enough_data", "min_rows": MIN_ROWS}
    psi = psi_per_feature(x, deciles)
    strong = psi > STRONG
    order = np.argsort(-psi)[:10]
    drifted = bool(strong.mean() > SHARE)
    return {
        "rows": len(x),
        "mean_psi": float(psi.mean()),
        "share_strong": float(strong.mean()),
        "status": "drifted"
        if drifted
        else ("moderate" if psi.mean() > 0.1 else "stable"),
        "drifted": drifted,
        "top": [{"feature": names[i], "psi": float(psi[i])} for i in order],
    }
