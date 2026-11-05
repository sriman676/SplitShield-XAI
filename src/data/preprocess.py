"""Authoritative preprocessing pipeline — the ten steps of 04_DATA §4, in order.

Order is not negotiable. Steps 3 (dedup), 6 (class-capped subsampling), 7
(split-before-scale), 8 (scaler fit on train only), and 9 (imbalance on train
only) are the ones that, if reordered, silently inflate every downstream number.

Each step logs a before/after row count (08_ENGINEERING_STANDARDS §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config.base import Config
from src.data import labels as L
from src.data.schema import validate_columns
from src.data.sources import DatasetSpec
from utils.logging_config import get_logger
from utils.paths import ensure_within

_log = get_logger(__name__)


@dataclass
class ProcessedData:
    """Result of preprocessing: scaled splits, labels, scaler, and provenance."""

    feature_names: list[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_names: tuple[str, ...]
    scaler: Any
    class_weights: np.ndarray | None
    frame: pd.DataFrame  # cleaned+subsampled (pre-scale) rows, for leakage checks
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


def _class_counts(y: np.ndarray, order: tuple[str, ...]) -> dict[str, int]:
    return {
        order[i]: int(n)
        for i, n in zip(*np.unique(y, return_counts=True), strict=False)
    }


def _subsample_positions(
    fam: np.ndarray, rare_floor: int, class_cap: int, seed: int
) -> np.ndarray:
    """Class-capped subsampling (NOT proportional) on row positions; rare classes
    kept whole. Same draw as ``DataFrame.groupby(...).sample(random_state=seed)``."""
    parts = []
    for f in sorted(set(fam.tolist())):
        pos = np.flatnonzero(fam == f)
        if len(pos) > rare_floor:
            n = min(len(pos), class_cap)
            pos = pd.Series(pos).sample(n=n, random_state=seed).to_numpy()
        parts.append(pos)
    return np.sort(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)


def preprocess_frame(
    df: pd.DataFrame,
    spec: DatasetSpec,
    cfg: Config,
    seed: int,
    run_directory: Path | None = None,
) -> ProcessedData:
    """Run the ten-step pipeline on an in-memory frame.

    Args:
        df: Raw (already column-stripped) frame.
        spec: Dataset registry entry.
        cfg: Resolved run config (subsample caps, scaler, imbalance strategy).
        seed: Controls subsample draw and the split (held constant across arms).
        run_directory: If given, persist scaler + parquet + split indices here.

    Returns:
        A ``ProcessedData`` with scaled splits and provenance stats.
    """
    stats: dict[str, Any] = {"n_rows_raw": len(df)}

    # 1. Schema validation (names already stripped by load_raw).
    validate_columns(df, spec.expected_columns)
    if spec.label_col not in df.columns:
        raise ValueError(f"Label column {spec.label_col!r} not found")

    # 2. Concatenate — done in load_raw; record the raw count above.

    # 3. Deduplicate exact duplicate rows (excluding any timestamp column).
    dedup_cols = [c for c in df.columns if c != spec.timestamp_col]
    before = len(df)
    # Row hashes, not drop_duplicates: pandas' multi-column group index needs several
    # GB on ~19M-row NetFlow frames and raised MemoryError. One uint64 per row instead.
    # ponytail: 64-bit hash collisions (~n^2/2^65, ~1e-5 at 19M rows) could drop a
    # unique row; use exact drop_duplicates if that ever matters more than memory.
    dup = (
        pd.util.hash_pandas_object(df[dedup_cols], index=False).duplicated().to_numpy()
    )
    stats["n_duplicates_dropped"] = int(dup.sum())
    _log.info(
        "Step 3 dedup: %d -> %d (dropped %d)",
        before,
        before - stats["n_duplicates_dropped"],
        stats["n_duplicates_dropped"],
    )

    # 4. Clean invalid values (NaN, +/-Inf from divide-by-zero flow rates).
    # Column by column on the underlying arrays (no full-frame replace/dropna copies,
    # which ran out of memory on the 2.8M-row original CIC-IDS2017); steps 3 and 4
    # are then applied as ONE filter. Same result as dedup-then-dropna: a row is kept
    # iff it is the first copy of its duplicates and has no NaN/Inf feature.
    # Step 5's identifier columns are decided first: they never become features, so
    # they must not veto rows here (a text column such as NetFlow's 'Attack' or
    # DIAD's 'Flow ID' coerces to all-NaN and used to drop EVERY row).
    to_drop = [c for c in spec.id_columns if c in df.columns]
    if not cfg.training.keep_dest_port:
        to_drop += [
            c
            for c in ("Destination Port", "Destination_Port", "Dst Port")
            if c in df.columns
        ]
    feat_cols = [c for c in df.columns if c != spec.label_col and c not in to_drop]
    bad = np.zeros(len(df), dtype=bool)
    bad_by_col: dict[str, int] = {}
    for c in feat_cols:
        v = df[c].to_numpy()
        if v.dtype.kind in "iub":
            continue  # integers / booleans cannot hold NaN or Inf
        if v.dtype.kind != "f":  # object column: "Infinity"/inf/text -> non-finite
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(
                np.float64, na_value=np.nan
            )
        m = ~np.isfinite(v)
        n = int((m & ~dup).sum())
        if n:
            bad_by_col[c] = n
            bad |= m
    stats["n_invalid_dropped"] = int((bad & ~dup).sum())
    # Memory: steps 3-6 only decide WHICH rows survive (masks + the label column);
    # the feature table is copied once, at the end, for the few hundred thousand
    # rows that remain. Copying the filtered multi-million-row frame (as before)
    # ran out of memory on the 2.5M-row CIC-IDS2017 V2 on a 16 GB machine.
    keep = np.flatnonzero(~(dup | bad))
    del dup, bad
    if bad_by_col:
        _log.info(
            "Step 4 clean: dropped %d rows; NaN/Inf by column: %s",
            stats["n_invalid_dropped"],
            bad_by_col,
        )

    # 5. Drop non-predictive identifiers, conditionally (list built before step 4).
    if to_drop:
        _log.info("Step 5 dropped identifier columns: %s", to_drop)
    stats["dropped_columns"] = to_drop

    # Drop configured raw labels (e.g. CIC-V2 'Comb': unmappable, uncertain
    # provenance) before mapping, with a logged count (04_DATA §3).
    raw_lbl = df[spec.label_col].iloc[keep].reset_index(drop=True)
    if spec.drop_labels:
        ok = ~raw_lbl.astype(str).str.strip().isin(spec.drop_labels).to_numpy()
        stats["n_dropped_labels"] = int((~ok).sum())
        keep, raw_lbl = keep[ok], raw_lbl[ok].reset_index(drop=True)
        _log.info(
            "Dropped %d rows with labels %s (spec.drop_labels)",
            stats["n_dropped_labels"],
            sorted(spec.drop_labels),
        )

    # Map labels to the active scheme (9-class family, or binary for E8).
    order = L.order_for(spec.label_scheme)
    fam = L.resolve_labels(raw_lbl, spec.label_scheme).to_numpy()
    del raw_lbl

    # 6. Class-capped subsampling (NOT proportional stratified).
    sel = _subsample_positions(
        fam, cfg.subsample.rare_floor, cfg.subsample.class_cap, seed
    )
    stats["n_after_subsample"] = len(sel)
    _log.info(
        "Step 6 subsample: %d -> %d (rare_floor=%d cap=%d)",
        len(keep),
        len(sel),
        cfg.subsample.rare_floor,
        cfg.subsample.class_cap,
    )
    rows = keep[sel]
    cols = [c for c in df.columns if c not in to_drop and c != spec.label_col]
    out = df.iloc[rows, [df.columns.get_loc(c) for c in cols]].reset_index(drop=True)
    out["_family"] = fam[sel]

    # Timestamp for a temporal split (captured from the raw column, which may be an
    # identifier dropped above). Excluded from the feature matrix.
    _TS = "__ts__"
    if cfg.split == "temporal":
        tcol = spec.timestamp_col
        if not tcol or tcol not in df.columns:
            raise ValueError(
                f"split='temporal' requires spec.timestamp_col present in the frame; "
                f"got timestamp_col={tcol!r} (04_DATA §4 step 7)."
            )
        from src.data.sources import parse_timestamps

        ts = parse_timestamps(df[tcol].iloc[rows], spec)
        if ts.isna().any():
            raise ValueError("split='temporal': unparseable timestamps present.")
        out[_TS] = ts.to_numpy()
    df = out
    del out

    y_all = L.encode_scheme(df["_family"], spec.label_scheme)
    feature_names = [c for c in df.columns if c not in ("_family", _TS)]
    X_all = df[feature_names].to_numpy(dtype=np.float64)

    # 7. 70/15/15 split, seeded, BEFORE scaling. Random is stratified on the 9-class
    # label; temporal is time-ordered (oldest->train, newest->test) with no leakage
    # across time and no stratify (class coverage is not guaranteed by construction).
    idx = np.arange(len(df))
    if cfg.split == "temporal":
        order_t = np.argsort(df[_TS].to_numpy(), kind="stable")
        n = len(order_t)
        n_test = max(1, round(0.15 * n))
        n_val = max(1, round(0.15 * n))
        idx_test = np.sort(order_t[n - n_test :])
        idx_val = np.sort(order_t[n - n_test - n_val : n - n_test])
        idx_tr = np.sort(order_t[: n - n_test - n_val])
        for name, part in (("train", idx_tr), ("val", idx_val), ("test", idx_test)):
            missing = set(np.unique(y_all)) - set(np.unique(y_all[part]))
            if missing:
                _log.warning(
                    "Temporal split: classes %s absent from %s (expected for "
                    "time-ordered NIDS splits, not a bug).",
                    sorted(missing),
                    name,
                )
    else:
        idx_tr, idx_test = train_test_split(
            idx, test_size=0.15, random_state=seed, stratify=y_all
        )
        idx_tr, idx_val = train_test_split(
            idx_tr, test_size=0.15 / 0.85, random_state=seed, stratify=y_all[idx_tr]
        )
        for name, part in (("train", idx_tr), ("val", idx_val), ("test", idx_test)):
            present = set(np.unique(y_all[part]))
            missing = set(range(len(order))) - present - _absent_classes(y_all, order)
            if missing:
                raise ValueError(
                    f"Stratified split failed: classes {sorted(missing)} absent from "
                    f"{name}. Every class present in the data must appear in all "
                    "splits (04_DATA §4 step 7)."
                )

    # 8. Scale — fit on train only, apply to val/test.
    scaler = MinMaxScaler() if cfg.training.scaler == "minmax" else StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_tr])
    X_val = scaler.transform(X_all[idx_val])
    X_test = scaler.transform(X_all[idx_test])
    y_train, y_val, y_test = y_all[idx_tr], y_all[idx_val], y_all[idx_test]

    # 9. Class imbalance — training split only, after scaling.
    class_weights: np.ndarray | None = None
    if cfg.training.imbalance == "smote":
        from imblearn.over_sampling import SMOTE  # guarded: heavy import

        X_train, y_train = SMOTE(random_state=seed).fit_resample(X_train, y_train)
    else:  # default: class-weighted loss
        classes = np.unique(y_train)
        w = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = np.ones(len(order), dtype=np.float64)
        for c, wc in zip(classes, w, strict=False):
            class_weights[c] = wc

    stats["class_counts"] = {
        "train": _class_counts(y_train, order),
        "val": _class_counts(y_val, order),
        "test": _class_counts(y_test, order),
    }

    # 10. Persist (scaler, parquet, split indices) if a run directory is given.
    if run_directory is not None:
        _persist(run_directory, spec, cfg, df, scaler, idx_tr, idx_val, idx_test)

    return ProcessedData(
        feature_names=feature_names,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_names=order,
        scaler=scaler,
        class_weights=class_weights,
        frame=df,
        idx_train=idx_tr,
        idx_val=idx_val,
        idx_test=idx_test,
        stats=stats,
    )


def _absent_classes(y_all: np.ndarray, order: tuple[str, ...]) -> set[int]:
    """Classes with zero samples in the whole dataset (cannot be split; not a bug)."""
    return set(range(len(order))) - set(np.unique(y_all))


def _persist(
    run_directory: Path,
    spec: DatasetSpec,
    cfg: Config,
    frame: pd.DataFrame,
    scaler: Any,
    idx_tr: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> None:
    """Save scaler + parquet + split indices under the run directory."""
    d = ensure_within(run_directory)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, d / "scaler.joblib")
    frame.to_parquet(d / "processed.parquet")
    np.savez(d / "split_indices.npz", train=idx_tr, val=idx_val, test=idx_test)
    _log.info("Persisted processed data for %s under %s", spec.id, d)
