"""Champion/challenger updates for the NetFlow and Zeek detectors (owner 2026-10-15).

The other models already have one (``src/mlareas/*.improve``,
``engine.retrain_from_feedback`` / ``add_dataset``, ``src/nlp``). Same rules:

  * new labelled data is mixed with the stored replay sample of the training split;
  * the challenger is judged on the FIXED validation sample saved next to the model
    (``holdout.npz``, built by ``scripts/build_holdouts.py``) - never on its own
    training rows and never on the test sample;
  * it replaces the champion only if validation F1 at the frozen alert threshold is
    not worse. The controller (``src/soc/controller.py``) adds the speed gate and
    keeps the snapshot for rollback.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

REPLAY_CAP = 40_000


def _f1(y: np.ndarray, alert: np.ndarray) -> float:
    tp = int((y.astype(bool) & alert).sum())
    fp = int((~y.astype(bool) & alert).sum())
    fn = int((y.astype(bool) & ~alert).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def _binary_labels(raw: pd.Series) -> np.ndarray:
    """1 = attack/malicious, 0 = benign; accepts 0/1 or names."""
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().all():
        return (num.to_numpy() > 0).astype(np.int8)
    s = raw.astype(str).str.strip().str.lower()
    return (~s.isin(["benign", "normal", "0", "false", "-"])).astype(np.int8).to_numpy()


def _load(folder: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    h, r = np.load(folder / "holdout.npz"), np.load(folder / "replay.npz")
    return {k: h[k] for k in h.files}, {k: r[k] for k in r.files}


# --- NetFlow -----------------------------------------------------------------------


def netflow_improve(df: pd.DataFrame, label_col: str = "Label") -> dict[str, Any]:
    """Refit the tree members (LightGBM + Random Forest) on replay + new rows; the
    MLP member, scaler, ensemble weights and threshold stay as deployed."""
    import joblib
    import torch

    from scripts.deploy_netflow import fit_trees
    from src.models.ensemble import EnsembleModel
    from src.soc import netflow_model as nm

    det = nm.load_netflow()
    if det is None:
        raise FileNotFoundError("NetFlow detector not deployed")
    hold, rep = _load(nm.NETFLOW_DIR)
    X_new, missing = det.align(df)
    y_new = _binary_labels(df[label_col])
    if len(missing) > len(det.feature_names) // 2:
        raise ValueError(f"too many NetFlow columns missing: {missing[:8]}")
    X_tr = np.vstack([rep["X"], X_new])
    y_tr = np.concatenate([rep["y"], y_new]).astype(int)

    def scaled(X: np.ndarray) -> np.ndarray:
        X = np.where(np.isfinite(X), X, det.medians)
        return np.asarray(det.scaler.transform(X), dtype=np.float32)

    benign = det.label_names.index("BENIGN")

    def alerts(scorer: Any, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            p = torch.softmax(scorer(torch.as_tensor(scaled(X))) / det.temperature, 1)
        return (1.0 - p.numpy()[:, benign]) >= det.alert_threshold

    trees = fit_trees(SimpleNamespace(X_train=scaled(X_tr), y_train=y_tr), seed=0)
    weights = det.meta.get("ensemble") or {"mlp": 1.0}
    challenger = EnsembleModel(det.model, trees, weights).eval()
    yv = hold["y_val"].astype(bool)
    f_c = _f1(yv, alerts(det.scorer, hold["X_val"]))
    f_n = _f1(yv, alerts(challenger, hold["X_val"]))
    promoted = f_n >= f_c
    if promoted:
        members = {k: v for k, v in trees.items() if k in weights}
        joblib.dump(members, nm.NETFLOW_DIR / "trees.joblib", compress=3)
        keep = slice(-REPLAY_CAP, None)
        np.savez_compressed(nm.NETFLOW_DIR / "replay.npz", X=X_tr[keep], y=y_tr[keep])
        nm._CACHE.clear()
    return {"promoted": promoted, "metric": "val_f1_at_threshold",
            "champion": round(f_c, 4), "challenger": round(f_n, 4),
            "new_rows": int(len(y_new)), "missing_columns": missing}  # fmt: skip


# --- Zeek --------------------------------------------------------------------------


def zeek_rows(path: Path) -> list[dict[str, Any]]:
    """A labelled Zeek conn.log (TSV with #fields; IoT-23 style ``label`` column).
    Parsed like scripts/extra_logs.py: IoT-23 joins the last header names and values
    with runs of spaces instead of tabs."""
    import re

    fields: list[str] = []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if raw.startswith("#fields"):
                fields = [
                    n
                    for c in raw.split("\t")[1:]
                    for n in re.split(r"\s{2,}", c.strip())
                ]
                continue
            if raw.startswith("#") or not raw.strip() or not fields:
                continue
            vals = raw.split("\t")
            if len(vals) < len(fields):
                vals = vals[:-1] + re.split(r"\s{2,}", vals[-1])
            out.append(dict(zip(fields, vals, strict=False)))
    return out


def zeek_improve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Refit the deployed model family on replay + new connections."""
    import joblib
    from sklearn.base import clone

    from src.soc import zeekml

    bundle = zeekml.load()
    if bundle is None:
        raise FileNotFoundError("Zeek model not trained")
    folder = Path(zeekml.MODEL).parent
    hold, rep = _load(folder)
    labelled = [r for r in rows if r.get("label") not in (None, "", "-")]
    if not labelled:
        raise ValueError("no labelled connections (need a 'label' field)")
    X_new = zeekml.matrix(labelled)
    y_new = np.array(
        [str(r["label"]).lower().startswith("malicious") for r in labelled], np.int8
    )
    X_tr = np.vstack([rep["X"], X_new]).astype(np.float32)
    y_tr = np.concatenate([rep["y"], y_new]).astype(int)
    chal = clone(bundle["model"]).fit(X_tr, y_tr)
    thr = float(bundle["threshold"])
    Xv, yv = hold["X_val"].astype(np.float32), hold["y_val"].astype(bool)
    f_c = _f1(yv, bundle["model"].predict_proba(Xv)[:, 1] >= thr)
    f_n = _f1(yv, chal.predict_proba(Xv)[:, 1] >= thr)
    promoted = f_n >= f_c
    if promoted:
        joblib.dump(dict(bundle, model=chal), zeekml.MODEL)
        keep = slice(-REPLAY_CAP, None)
        np.savez_compressed(folder / "replay.npz", X=X_tr[keep], y=y_tr[keep])
        zeekml._LOADED.clear()
    return {"promoted": promoted, "metric": "val_f1_at_threshold",
            "champion": round(f_c, 4), "challenger": round(f_n, 4),
            "new_rows": int(len(y_new))}  # fmt: skip
