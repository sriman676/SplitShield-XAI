"""Data-quality report for every dataset with data (owner request 2026-10-14).

Per dataset, on the raw rows the pipeline loads (up to ``MAX_ROWS``, sampled
evenly): size, class balance, exact duplicates, conflicting labels (identical
features, different label - label noise), NaN / Inf by column, negative values in
columns that cannot be negative (counts, durations, lengths), constant and
near-constant columns, duplicated columns, identifier columns (leakage risk), and
whether the release is already min-max normalized. Read-only: nothing is changed;
the pipeline's own cleaning steps are listed next to what they handle.

Writes results/data_quality.json (rendered in FINAL_REPORT).

    python -m scripts.data_quality [--max-rows 1000000]
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data_quality.json"
MAX_ROWS = 1_000_000
ID_HINT = re.compile(
    r"flow.?id|src.?ip|dst.?ip|source.?ip|destination.?ip|timestamp|^ip", re.I
)
NONNEG = re.compile(
    r"duration|packets|bytes|length|count|size|iat|total|subflow|win|active|idle", re.I
)
HANDLED = {
    "duplicates": "pipeline step 3 removes exact duplicate rows",
    "nan_inf": "pipeline step 4 drops rows with NaN/Inf",
    "identifiers": "pipeline step 5 drops identifier columns",
    "imbalance": "class-capped subsample (rare floor 5k, cap 60k) + class weights",
}


def profile(df: pd.DataFrame, label: str) -> dict[str, Any]:
    """Quality profile of one raw frame (``label`` = the label column)."""
    cols = [c for c in df.columns if c != label]
    num_all = df[cols].apply(pd.to_numeric, errors="coerce")
    # text columns (IPs, timestamps, attack names) are not numeric features: their
    # values would all count as NaN and flag every row
    text = [c for c in cols if df[c].dtype == object and num_all[c].isna().mean() > 0.5]
    feats = [c for c in cols if c not in text]
    num = num_all[feats]
    arr = num.to_numpy(np.float64)
    n = len(df)
    y = df[label].astype(str)
    counts = y.value_counts()

    finite = np.isfinite(arr)
    nan_by = {c: int(v) for c, v in zip(feats, np.isnan(arr).sum(0), strict=True) if v}
    inf_by = {c: int(v) for c, v in zip(feats, np.isinf(arr).sum(0), strict=True) if v}

    h = pd.util.hash_pandas_object(num, index=False)
    dup_rows = int(pd.DataFrame({"h": h, "y": y.values}).duplicated().sum())
    by_h = pd.DataFrame({"h": h.values, "y": y.values}).groupby("h")["y"].nunique()
    conflict_groups = by_h[by_h > 1]
    conflict_rows = int(h.isin(conflict_groups.index).sum())

    const, near = [], []
    for j, c in enumerate(feats):
        col = arr[:, j][finite[:, j]]
        if col.size == 0:
            continue
        vals, cts = np.unique(col, return_counts=True)
        if len(vals) == 1:
            const.append(c)
        elif cts.max() / col.size > 0.999:
            near.append(c)
    col_hash: dict[int, str] = {}
    dup_cols = []
    for c in feats:
        k = int(pd.util.hash_pandas_object(num[c], index=False).sum())
        if k in col_hash:
            dup_cols.append([col_hash[k], c])
        else:
            col_hash[k] = c
    neg = {
        c: int(v)
        for c, v in zip(feats, (np.where(finite, arr, 0) < 0).sum(0), strict=True)
        if v and NONNEG.search(c)
    }
    with np.errstate(invalid="ignore"):
        lo, hi = (
            np.nanmin(np.where(finite, arr, np.nan)),
            np.nanmax(np.where(finite, arr, np.nan)),
        )
    return {
        "rows": n,
        "features": len(feats),
        "classes": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": float(counts.max() / max(1, counts.min())),
        "duplicate_rows": dup_rows,
        "duplicate_share": dup_rows / max(1, n),
        "conflicting_label_rows": conflict_rows,
        "conflicting_label_share": conflict_rows / max(1, n),
        "nan_by_column": nan_by,
        "inf_by_column": inf_by,
        "rows_with_nan_or_inf": int((~finite).any(1).sum()),
        "negative_in_nonnegative_columns": neg,
        "constant_columns": const,
        "near_constant_columns": near,
        "duplicated_columns": dup_cols,
        "identifier_columns": [c for c in cols if ID_HINT.search(c)],
        "text_columns": text,
        "prenormalized_0_1": bool(lo >= 0 and hi <= 1),
    }


def findings(p: dict[str, Any]) -> list[str]:
    """Plain-language problems worth knowing, most important first."""
    out = []
    if p["prenormalized_0_1"]:
        out.append("release is min-max normalized to [0,1]: raw units unrecoverable")
    if p["conflicting_label_share"] > 0.001:
        out.append(
            f"{p['conflicting_label_share']:.2%} of rows share identical features with "
            "a differently labelled row (label noise / ambiguous flows)"
        )
    if p["negative_in_nonnegative_columns"]:
        cols = ", ".join(list(p["negative_in_nonnegative_columns"])[:4])
        out.append(f"negative values in columns that cannot be negative: {cols}")
    if p["duplicate_share"] > 0.01:
        out.append(
            f"{p['duplicate_share']:.1%} exact duplicate rows (removed by step 3)"
        )
    if p["rows_with_nan_or_inf"]:
        out.append(f"{p['rows_with_nan_or_inf']} rows with NaN/Inf (removed by step 4)")
    if p["constant_columns"]:
        out.append(
            f"{len(p['constant_columns'])} constant columns (carry no information)"
        )
    if p["duplicated_columns"]:
        out.append(f"{len(p['duplicated_columns'])} duplicated column pair(s)")
    if p["imbalance_ratio"] > 1000:
        out.append(
            f"extreme class imbalance (largest/smallest {p['imbalance_ratio']:.0f}x)"
        )
    return out


def run(max_rows: int = MAX_ROWS) -> dict[str, Any]:
    from src.data.sources import REGISTRY, load_raw

    res: dict[str, Any] = {
        "max_rows": max_rows,
        "handled_by_pipeline": HANDLED,
        "datasets": {},
    }
    for ds, spec in REGISTRY.items():
        if ds == "fixture" or not (ROOT / "data" / "raw" / ds).is_dir():
            continue
        try:
            df = load_raw(spec)
        except (FileNotFoundError, ValueError, OSError) as exc:
            res["datasets"][ds] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        total = len(df)
        if total > max_rows:  # evenly spaced rows: every file/day represented
            df = df.iloc[np.linspace(0, total - 1, max_rows).astype(int)]
        p = profile(df, spec.label_col)
        p["rows_total"] = total
        p["findings"] = findings(p)
        res["datasets"][ds] = p
        print(
            f"[data-quality] {ds}: {total} rows; "
            + ("; ".join(p["findings"]) or "no issues")
        )
        del df
        gc.collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), "utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS)
    run(ap.parse_args().max_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
