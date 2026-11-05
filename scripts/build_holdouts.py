"""Save fixed held-out + replay samples next to the NetFlow and Zeek models (owner
2026-10-15), so their champion/challenger updates still work after the datasets are
removed (pendrive copy).

Per model, in raw units (what the live detector receives):
  holdout.npz  X_val, y_val    -> the GATE (selection on validation only)
               X_test, y_test  -> reported, never used to decide
  replay.npz   X, y            -> stratified sample of the training split, mixed
                                  with new labelled data when a challenger is fitted

Splits are rebuilt exactly as when the models were trained (same seeds / captures).

    python -m scripts.build_holdouts [--only netflow,zeek]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT

_log = get_logger(__name__)
CAP = {"val": 30_000, "test": 30_000, "replay": 40_000}


def _strat(
    X: np.ndarray, y: np.ndarray, n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """At most ``n`` rows, class shares kept (every class keeps >= 1 row)."""
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(1, round(n * len(idx) / len(y)))
        keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
    sel = np.sort(np.concatenate(keep))
    return X[sel], y[sel]


def _save(folder: Path, splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    (Xv, yv), (Xt, yt), (Xr, yr) = (
        _strat(*splits["val"], CAP["val"]),
        _strat(*splits["test"], CAP["test"]),
        _strat(*splits["train"], CAP["replay"]),
    )
    np.savez_compressed(folder / "holdout.npz", X_val=Xv.astype(np.float64), y_val=yv,
                        X_test=Xt.astype(np.float64), y_test=yt)  # fmt: skip
    np.savez_compressed(folder / "replay.npz", X=Xr.astype(np.float64), y=yr)
    _log.info(
        "%s: val %d, test %d, replay %d rows", folder.name, len(yv), len(yt), len(yr)
    )


def netflow() -> None:
    from scripts import deploy_netflow as dn
    from src.soc.netflow_model import NETFLOW_DIR

    (data, _cfg), _ = dn.source_data(PROJECT_ROOT, seed=0)
    raw = dn.raw_splits(data)
    ys = [np.asarray(data.y_train), np.asarray(data.y_val), np.asarray(data.y_test)]
    if any(len(r) != len(y) for r, y in zip(raw, ys, strict=True)):  # e.g. SMOTE rows
        raise RuntimeError(
            "raw rows and labels differ in length; replay would be mislabelled"
        )
    _save(
        NETFLOW_DIR,
        {s: (raw[i], ys[i]) for i, s in enumerate(("train", "val", "test"))},
    )


def zeek() -> None:
    from scripts import extra_logs as el
    from src.soc import zeekml

    splits = {}
    for split, caps in el.ZEEK_SPLIT.items():
        rows = [
            r
            for c in caps
            if (el.RAW_DIR / "iot-23" / c).is_dir()
            for r in el._zeek_rows(c)
        ]
        present = [c for c in caps if any(r["_capture"] == c for r in rows)]
        X = np.vstack(
            [zeekml.matrix([r for r in rows if r["_capture"] == c]) for c in present]
        )
        y = np.array([r["_malicious"] for c in present for r in rows if r["_capture"] == c],
                     dtype=np.int8)  # fmt: skip
        splits[split] = (X, y)
    _save(Path(zeekml.MODEL).parent, splits)


def flow() -> None:
    """Speed / sanity probe for the deployed CICFlowMeter detector: 5,000 raw rows
    (stratified over labels) from CIC-IDS2017 original, aligned to its features.
    Speed is the only thing gated on it (its quality gate is in engine._finetune)."""
    import glob

    import pandas as pd

    from src.soc import engine
    from src.soc.columns import align

    det = engine.load_detector()
    if det is None:
        raise RuntimeError("deployed detector missing")
    files = sorted(glob.glob(str(PROJECT_ROOT / "data/raw/cicids2017-original/*.csv")))
    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False, nrows=200_000)
        d.columns = [c.strip() for c in d.columns]
        parts.append(d.sample(min(len(d), 2_000), random_state=0))
    df = pd.concat(parts, ignore_index=True)
    X_df, _ = align(df, det.feature_names)
    X = (
        X_df[det.feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    y = df["Label"].astype(str).to_numpy()
    Xs, ys = _strat(X, y, 5_000)
    np.savez_compressed(
        PROJECT_ROOT / "artifacts" / "deploy" / "probe.npz", X=Xs, y=ys.astype(str)
    )
    _log.info("deploy: probe %d rows", len(ys))


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="flow,netflow,zeek")
    for name in ap.parse_args().only.split(","):
        {"flow": flow, "netflow": netflow, "zeek": zeek}[name]()


if __name__ == "__main__":
    main()
