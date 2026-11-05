"""Insider threat / user-behaviour analytics (UEBA) for the SOC (owner 2026-10-15).

Purpose: score each user-day of activity (logons, USB use, file copies to
removable media, email, web) for insider-threat risk and explain the score in
plain English for a non-expert analyst. Trained on CMU CERT r4.2 (dense-needle
release, 70 insiders); CMU CERT r4.1 is reserved as the never-seen blind set and
is only ever touched through ``blind_test`` (frozen model and threshold).

SOC API
  * ``score_user_day(features)`` -> ``{"risk", "verdict", "why"}``; ``why`` is a
    numbered list of checkable facts taken from the given features, then the
    verdict and what to do.
  * ``improve(new_labelled_data)`` -> champion/challenger on the fixed validation
    set; the challenger is kept only if validation PR-AUC is not worse; the old
    model stays under ``prev/`` for rollback; speed is recorded.
  * ``blind_test(release_dir)`` -> metrics of the frozen model on an unseen
    release (not run during development).

Rules: model and threshold are chosen on VALIDATION only; test is reported.
Models live under ``artifacts/mlareas/insider/{current,prev}`` of the main
project (override with ``MLAREAS_ARTIFACTS``).
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.mlareas import insider_features as feats

AREA = "insider"
FAMILIES = ("lightgbm", "isoforest", "logreg")
TREE_FAMILIES = ("lightgbm", "isoforest")
GAP_RULE = 0.05
SPLIT_FRACS = (0.70, 0.15)  # train, validation; test = rest
LGB_GRID: tuple[dict[str, Any], ...] = (
    {"num_leaves": 15, "min_child_samples": 20},
    {"num_leaves": 31, "min_child_samples": 50},
    {"num_leaves": 63, "min_child_samples": 100},
)
BUDGET_K = 5  # analyst budget: top-k user-days reviewed per day


# ------------------------------------------------------------------- paths


def main_project_root() -> Path:
    """The main checkout root, even when this code runs from a git worktree."""
    here = Path(__file__).resolve().parents[2]
    parts = here.parts
    if ".claude" in parts:
        return Path(*parts[: parts.index(".claude")])
    return here


def area_dir(root: Path | None = None) -> Path:
    """Model store of this area (``MLAREAS_ARTIFACTS`` overrides the root)."""
    if root is None:
        env = os.environ.get("MLAREAS_ARTIFACTS")
        root = Path(env) if env else main_project_root() / "artifacts" / "mlareas"
    return Path(root) / AREA


# ------------------------------------------------------------------ helpers


def mean_sd(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "sd": float("nan")}
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(statistics.fmean(vals), 4), "sd": round(sd, 4)}


def fit_verdict(train: float, val: float, tree_model: bool = False) -> dict[str, Any]:
    """Project gap rule: train - val > 0.05 -> overfitting."""
    gap = train - val
    if gap > GAP_RULE:
        verdict = "overfitting"
    elif train < 0.5 and val < 0.5:
        verdict = "underfitting"
    else:
        verdict = "good fit"
    note = ""
    if tree_model and gap > GAP_RULE:
        note = "tree ensembles fit train near-perfectly by design; judge on val vs test"
    return {
        "train": round(train, 4),
        "val": round(val, 4),
        "gap": round(gap, 4),
        "verdict": verdict,
        "note": note,
    }


def items_per_second(fn: Callable[[], Any], n_items: int, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return round(n_items / max(best, 1e-9), 1)


def _slog(x: np.ndarray) -> np.ndarray:
    """Signed log1p: tames heavy-tailed counts for the linear model."""
    return np.sign(x) * np.log1p(np.abs(x))


# ------------------------------------------------------------------- splits


def time_split(days: pd.Series, fracs: tuple[float, float] = SPLIT_FRACS) -> pd.Series:
    """train = earliest 70% of calendar days, val = next 15%, test = last 15%."""
    uniq = np.sort(days.unique())
    n = len(uniq)
    t_end = uniq[int(n * fracs[0]) - 1]
    v_end = uniq[int(n * (fracs[0] + fracs[1])) - 1]
    out = np.where(days <= t_end, "train", np.where(days <= v_end, "val", "test"))
    return pd.Series(out, index=days.index)


def user_split(
    users: pd.Series,
    insider_users: set[str],
    seed: int,
    fracs: tuple[float, float] = SPLIT_FRACS,
) -> pd.Series:
    """Split by USER (stratified on insider status): test users never trained on."""
    rng = np.random.default_rng(seed)
    assign: dict[str, str] = {}
    uniq = sorted(users.unique())
    for group in (
        [u for u in uniq if u in insider_users],
        [u for u in uniq if u not in insider_users],
    ):
        g = list(rng.permutation(group))
        n_tr = int(round(len(g) * fracs[0]))
        n_va = int(round(len(g) * fracs[1]))
        for i, u in enumerate(g):
            assign[u] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return users.map(assign)


# ------------------------------------------------------------------- models


def make_model(family: str, seed: int, params: Mapping[str, Any] | None = None) -> Any:
    """Unfitted model of one family."""
    params = dict(params or {})
    if family == "lightgbm":
        defaults: dict[str, Any] = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "class_weight": "balanced",
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": 4,
            "verbose": -1,
            "metric": "average_precision",  # early stopping watches PR-AUC only
        }
        return lgb.LGBMClassifier(**{**defaults, **params})
    if family == "isoforest":
        return IsolationForest(
            n_estimators=300, max_samples=8192, random_state=seed, n_jobs=4
        )
    if family == "logreg":
        return Pipeline(
            [
                ("slog", FunctionTransformer(_slog)),
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        class_weight="balanced", C=params.get("C", 0.1), max_iter=2000
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown family {family}")


def model_scores(family: str, model: Any, X: np.ndarray) -> np.ndarray:
    """Higher score = more suspicious, for every family."""
    if family == "isoforest":
        return np.asarray(-model.score_samples(X), dtype=np.float64)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def fit_family(
    family: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Fit one family; small hyper-parameter choice on VALIDATION PR-AUC."""
    if family == "isoforest":
        m = make_model(family, seed)
        m.fit(X_tr)  # unsupervised: labels are never used
        return m, {}
    grid: Sequence[dict[str, Any]]
    grid = LGB_GRID if family == "lightgbm" else ({"C": 0.01}, {"C": 0.1}, {"C": 1.0})
    best: tuple[float, Any, dict[str, Any]] = (-1.0, None, {})
    for params in grid:
        m = make_model(family, seed, params)
        if family == "lightgbm":
            m.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(100, first_metric_only=True, verbose=False)
                ],
            )
            params = {**params, "n_estimators": int(m.best_iteration_ or 800)}
        else:
            m.fit(X_tr, y_tr)
        ap = _pr_auc(y_va, model_scores(family, m, X_va))
        if ap > best[0]:
            best = (ap, m, params)
    return best[1], best[2]


# ------------------------------------------------------------------ metrics


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def choose_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold that maximises F1 (used on validation only)."""
    if y.sum() == 0:
        return float(np.quantile(s, 0.999))
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def point_metrics(
    y: np.ndarray, s: np.ndarray, thr: float, n_days: int
) -> dict[str, float]:
    """Precision / recall / F1 / PR-AUC / ROC-AUC and the alert load."""
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and np.isfinite(rec) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(_pr_auc(y, s), 4),
        "roc_auc": round(_roc_auc(y, s), 4),
        "alerts": int(pred.sum()),
        "alerts_per_day": round(float(pred.sum()) / max(n_days, 1), 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pos": int(y.sum()),
        "n_rows": int(len(y)),
    }


def recall_at_budget(df: pd.DataFrame, s: np.ndarray, k: int = BUDGET_K) -> float:
    """Recall of malicious user-days if the analyst reviews the top-k per day."""
    tmp = pd.DataFrame(
        {"day": df["day"].to_numpy(), "s": s, "y": df["label"].to_numpy()}
    )
    tmp["rank"] = tmp.groupby("day")["s"].rank(ascending=False, method="first")
    pos = tmp["y"].sum()
    return (
        round(float(tmp.loc[tmp["rank"] <= k, "y"].sum() / pos), 4)
        if pos
        else float("nan")
    )


def user_level(df: pd.DataFrame, s: np.ndarray, thr: float) -> list[dict[str, Any]]:
    """Per insider with malicious days in ``df``: detected? how early/late?"""
    tmp = df[["user", "day", "label", "insider"]].assign(s=s, alert=s >= thr)
    out = []
    for user, g in tmp[tmp["insider"] == 1].groupby("user"):
        mal = g[g["label"] == 1]
        if mal.empty:
            continue
        first_mal = mal["day"].min()
        hit = mal[mal["alert"]]
        any_alert = g[g["alert"]]
        out.append(
            {
                "user": str(user),
                "first_malicious_day": str(first_mal.date()),
                "malicious_days": int(len(mal)),
                "detected": bool(len(hit) > 0),
                "malicious_days_alerted": int(len(hit)),
                "delay_days": int((hit["day"].min() - first_mal).days)
                if len(hit)
                else None,
                "first_alert_vs_first_malicious_days": (
                    int((any_alert["day"].min() - first_mal).days)
                    if len(any_alert)
                    else None
                ),
                "false_alert_days": int((g["alert"] & (g["label"] == 0)).sum()),
                "max_risk": round(float(g["s"].max()), 4),
            }
        )
    return out


def evaluate_split(
    df: pd.DataFrame, X: np.ndarray, family: str, model: Any, thr: float
) -> dict[str, Any]:
    s = model_scores(family, model, X)
    y = df["label"].to_numpy()
    m = point_metrics(y, s, thr, df["day"].nunique())
    m["recall_at_budget"] = recall_at_budget(df, s, BUDGET_K)
    return {"metrics": m, "scores": s}


# --------------------------------------------------------------- experiment


def run_experiment(
    df: pd.DataFrame,
    split: pd.Series,
    seed: int,
    features: Sequence[str] | None = None,
    families: Sequence[str] = FAMILIES,
) -> dict[str, Any]:
    """Fit every family on train, pick params+threshold on val, report all splits."""
    cols = list(features or feats.feature_columns())
    parts = {k: df[split == k] for k in ("train", "val", "test")}
    Xs = {k: v[cols].to_numpy(np.float32) for k, v in parts.items()}
    ys = {k: v["label"].to_numpy() for k, v in parts.items()}
    res: dict[str, Any] = {"seed": seed, "families": {}}
    for fam in families:
        t0 = time.perf_counter()
        model, params = fit_family(
            fam, Xs["train"], ys["train"], Xs["val"], ys["val"], seed
        )
        fit_s = time.perf_counter() - t0
        s_val = model_scores(fam, model, Xs["val"])
        thr = choose_threshold(ys["val"], s_val)
        out: dict[str, Any] = {
            "params": params,
            "threshold": thr,
            "fit_seconds": round(fit_s, 2),
        }
        for k in ("train", "val", "test"):
            ev = evaluate_split(parts[k], Xs[k], fam, model, thr)
            out[k] = ev["metrics"]
            if k == "test":
                out["users_test"] = user_level(parts[k], ev["scores"], thr)
                out["test_scores"] = ev["scores"]
            if k == "val":
                out["users_val"] = user_level(parts[k], ev["scores"], thr)
        X_te = Xs["test"]

        def _score_test(f: str = fam, m: Any = model, X: np.ndarray = X_te) -> Any:
            return model_scores(f, m, X)

        out["rows_per_s"] = items_per_second(_score_test, len(X_te))
        out["model"] = model
        res["families"][fam] = out
    best = max(
        families,
        key=lambda f: np.nan_to_num(res["families"][f]["val"]["pr_auc"], nan=-1.0),
    )
    res["selected"] = best
    return res


# ------------------------------------------------------------- model store


def save_bundle(bundle: dict[str, Any], root: Path | None = None) -> Path:
    """Save as current model, moving the old one to ``prev/`` (rollback)."""
    base = area_dir(root)
    cur, prev = base / "current", base / "prev"
    if cur.exists():
        if prev.exists():
            shutil.rmtree(prev)  # our own older model copy only, never data
        cur.rename(prev)
    cur.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cur / "model.joblib")
    meta = {
        k: v
        for k, v in bundle.items()
        if isinstance(v, str | int | float | dict | list)
    }
    (cur / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return cur


def load_bundle(root: Path | None = None) -> dict[str, Any]:
    path = area_dir(root) / "current" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no trained insider model at {path}; run python -m scripts.train_insider"
        )
    bundle: dict[str, Any] = joblib.load(path)
    return bundle


def make_bundle(
    family: str,
    model: Any,
    params: Mapping[str, Any],
    features: Sequence[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_train: np.ndarray,
) -> dict[str, Any]:
    """Frozen deployable package: model + val threshold + fill values + speed."""
    s_val = model_scores(family, model, X_val)
    thr = choose_threshold(y_val, s_val)
    benign = s_val[y_val == 0] if (y_val == 0).any() else s_val
    medium = float(min(np.quantile(benign, 0.99), thr))
    return {
        "area": AREA,
        "family": family,
        "model": model,
        "params": dict(params),
        "features": list(features),
        "threshold": thr,
        "medium_threshold": medium,
        "val_pr_auc": round(_pr_auc(y_val, s_val), 4),
        "fill": {
            c: float(v)
            for c, v in zip(features, np.median(X_train, axis=0), strict=True)
        },
        "rows_per_s": items_per_second(
            lambda: model_scores(family, model, X_val), len(X_val)
        ),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_store_data(
    train: pd.DataFrame, val: pd.DataFrame, root: Path | None = None
) -> None:
    """Keep the fixed train / validation sets for champion/challenger updates."""
    d = area_dir(root) / "data"
    d.mkdir(parents=True, exist_ok=True)
    train.to_parquet(d / "train.parquet", index=False)
    val.to_parquet(d / "val.parquet", index=False)


def _load_store_data(root: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = area_dir(root) / "data"
    return pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "val.parquet")


# ---------------------------------------------------------------- SOC: score

_GROUPS: dict[str, str] = {
    "usb_conn_n": "usb", "usb_conn_ah": "usb", "usb_disc_n": "usb", "usb_days7": "usb",
    "file_n": "file", "file_ah": "file", "file_doc": "file", "file_exe": "file",
    "file_zip": "file", "file_img": "file",
    "http_job": "job", "job_days7": "job",
    "http_wikileaks": "wikileaks", "http_keylog": "keylog", "http_cloud": "cloud",
    "http_n": "web", "http_ah": "web",
    "logon_n": "hours", "logoff_n": "hours", "logon_ah": "hours", "ah_days7": "hours",
    "first_hour": "hours", "last_hour": "hours", "weekday": "hours", "is_weekend": "hours",
    "pc_n": "pcs",
    "email_n": "email", "email_ext_n": "email", "email_recip_n": "email",
    "email_bytes": "email", "email_attach_n": "email", "email_ext_attach_n": "email",
    "email_ah": "email",
    "hr_leaving_soon": "leaving",
    "O": "profile", "C": "profile", "E": "profile", "A": "profile", "N": "profile",
    "role_code": "profile", "dept_code": "profile", "func_code": "profile",
}  # fmt: skip

_VERDICTS: dict[str, tuple[str, str]] = {
    "usb": ("data theft via a USB / removable drive",
            "review which files were copied and to which device, talk to HR/legal, "
            "and check outbound email and web uploads for the same files"),
    "file": ("data theft via copying files to removable media",
             "review the copied files (sensitivity, volume), talk to HR/legal, "
             "and check whether the user is leaving"),
    "wikileaks": ("leaking data to WikiLeaks",
                  "preserve the web-proxy and USB logs, escalate to legal / CISO "
                  "immediately, and check what was uploaded"),
    "keylog": ("credential theft or sabotage using a keylogger",
               "isolate the machines involved, scan for keylogging software, reset "
               "the passwords of the user and their colleagues, escalate to the CISO"),
    "job": ("a departing employee taking data with them",
            "check USB and email activity for data leaving, and confirm the "
            "user's status with HR"),
    "leaving": ("a leaver taking data with them",
                "confirm the leaving date with HR, review recent USB, file and email "
                "activity, and plan access removal"),
    "email": ("data exfiltration over email",
              "review recipients outside the company and the attachments, talk to "
              "the user's manager"),
    "pcs": ("use of other people's computers (possible masquerading)",
            "check whose machines were used and why, and look for logins with "
            "someone else's account"),
    "hours": ("unusual working hours",
              "confirm with the manager whether the after-hours work was expected"),
    "cloud": ("uploading data to a personal cloud service",
              "check what was uploaded and to which account"),
    "web": ("unusual web activity", "review the user's web-proxy log for the day"),
    "profile": ("a static profile pattern (role / personality survey)",
                "no action from this alone; look for supporting activity"),
}  # fmt: skip


def _base(col: str) -> str:
    for suffix in ("_mean30", "_z30", "_peerz"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def _contributions(bundle: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    """Per-feature push towards 'malicious' for one row (same order as features)."""
    fam, model = bundle["family"], bundle["model"]
    if fam == "lightgbm":
        contrib = model.booster_.predict(x, pred_contrib=True)
        return np.asarray(contrib[0, :-1], dtype=np.float64)
    if fam == "logreg":
        z = model[:-1].transform(x)
        return np.asarray(z[0] * model[-1].coef_[0], dtype=np.float64)
    fill = np.array([bundle["fill"][c] for c in bundle["features"]])
    return np.abs(x[0] - fill) / (np.abs(fill) + 1.0)  # isoforest: deviation size


def _num(f: Mapping[str, Any], key: str) -> float | None:
    v = f.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _vs_normal(f: Mapping[str, Any], key: str) -> str:
    """'— 30x this user's normal' when the baseline mean is supplied."""
    v, m = _num(f, key), _num(f, f"{key}_mean30")
    if v is None or m is None:
        return ""
    if m <= 0 and v > 0:
        return " - this user normally never does this (none in the previous 30 days)"
    if m > 0 and v / m >= 2:
        return f" - {v / m:.0f}x this user's normal (30-day average {m:.1f})"
    return ""


def _peer(f: Mapping[str, Any], key: str) -> str:
    z = _num(f, f"{key}_peerz")
    return (
        f" and {z:.1f} standard deviations above colleagues in the same role"
        if z and z >= 2
        else ""
    )


def _fact(group: str, f: Mapping[str, Any]) -> str | None:
    """One plain-English, checkable sentence per activity group (None if unsupported)."""
    n = _num
    if group == "usb" and n(f, "usb_conn_n") is not None:
        ah = n(f, "usb_conn_ah") or 0
        s = f"Connected a USB / removable drive {n(f, 'usb_conn_n'):.0f} times today"
        s += f" ({ah:.0f} after hours)" if ah else ""
        s += _vs_normal(f, "usb_conn_n") + _peer(f, "usb_conn_n")
        if n(f, "usb_days7"):
            s += f"; USB use on {n(f, 'usb_days7'):.0f} of the last 7 days"
        return s + "."
    if group == "file" and n(f, "file_n") is not None:
        ah = n(f, "file_ah") or 0
        kinds = [
            f"{n(f, k):.0f} {lbl}"
            for k, lbl in (
                ("file_doc", "documents"),
                ("file_exe", "programs"),
                ("file_zip", "archives"),
            )
            if n(f, k)
        ]
        s = f"Copied {n(f, 'file_n'):.0f} files to removable media"
        s += (
            f", {ah:.0f} of them after hours (outside 07:00-19:00 or weekend)"
            if ah
            else ""
        )
        s += f" ({', '.join(kinds)})" if kinds else ""
        return s + _vs_normal(f, "file_n") + _peer(f, "file_n") + "."
    if group == "job" and (
        n(f, "http_job") is not None or n(f, "job_days7") is not None
    ):
        s = f"Visited job-search sites {n(f, 'http_job') or 0:.0f} times today"
        if n(f, "job_days7") is not None:
            s += f"; job-search activity on {n(f, 'job_days7'):.0f} of the last 7 days"
        return s + _vs_normal(f, "http_job") + "."
    if group == "wikileaks" and n(f, "http_wikileaks"):
        return f"Visited WikiLeaks {n(f, 'http_wikileaks'):.0f} times today{_vs_normal(f, 'http_wikileaks')}."
    if group == "keylog" and n(f, "http_keylog"):
        return f"Visited keylogger / spyware download sites {n(f, 'http_keylog'):.0f} times today."
    if group == "cloud" and n(f, "http_cloud"):
        return f"Visited cloud file-sharing sites {n(f, 'http_cloud'):.0f} times today."
    if group == "web" and n(f, "http_n") is not None:
        ah = n(f, "http_ah") or 0
        s = f"Made {n(f, 'http_n'):.0f} web requests"
        s += f" ({ah:.0f} after hours)" if ah else ""
        return s + _vs_normal(f, "http_n") + "."
    if group == "hours" and n(f, "logon_ah") is not None:
        s = f"Logged on {n(f, 'logon_ah'):.0f} times outside 07:00-19:00 or at the weekend"
        if n(f, "ah_days7"):
            s += f"; after-hours logons on {n(f, 'ah_days7'):.0f} of the last 7 days"
        fh = n(f, "first_hour")
        if fh is not None and fh >= 0:
            s += f"; first logon at {fh:02.0f}:00"
        return s + _vs_normal(f, "logon_ah") + "."
    if group == "pcs" and n(f, "pc_n") is not None:
        return f"Used {n(f, 'pc_n'):.0f} different computers today{_vs_normal(f, 'pc_n')}{_peer(f, 'pc_n')}."
    if group == "email" and n(f, "email_n") is not None:
        s = f"Sent {n(f, 'email_n'):.0f} emails"
        if n(f, "email_ext_n"):
            s += f", {n(f, 'email_ext_n'):.0f} to addresses outside the company"
        if n(f, "email_ext_attach_n"):
            s += f" carrying {n(f, 'email_ext_attach_n'):.0f} attachments"
        if n(f, "email_bytes"):
            s += f" ({(n(f, 'email_bytes') or 0) / 1e6:.1f} MB in total)"
        return s + _vs_normal(f, "email_ext_n") + "."
    if group == "leaving" and n(f, "hr_leaving_soon"):
        return "HR records show this user leaves the organisation within about a month."
    if group == "profile":
        return "Static profile fields (role, department, personality survey) add a little to the score."
    return None


_GENERIC = ("web", "hours")  # verdict prefers a specific threat when one is supported


def _verdict_word(risk: float, bundle: Mapping[str, Any]) -> str:
    if risk >= bundle["threshold"]:
        return "high"
    if risk >= bundle["medium_threshold"]:
        return "medium"
    return "low"


def explain(
    bundle: Mapping[str, Any], features: Mapping[str, Any], risk: float, x: np.ndarray
) -> list[str]:
    """Numbered plain-English reasons, then verdict and what to do."""
    cols = bundle["features"]
    contrib = _contributions(bundle, x)
    groups: dict[str, float] = {}
    for c, v in zip(cols, contrib, strict=True):
        g = _GROUPS.get(_base(c))
        if g and c in features:
            groups[g] = groups.get(g, 0.0) + float(v)
    ranked = [g for g, v in sorted(groups.items(), key=lambda kv: -kv[1]) if v > 0]
    lines: list[str] = []
    for g in ranked:
        fact = _fact(g, features)
        if fact:
            lines.append(fact)
        if len(lines) == 4:
            break
    level = _verdict_word(risk, bundle)
    supported = [g for g in ranked if g != "profile" and _fact(g, features)]
    specific = [g for g in supported if g not in _GENERIC]
    choices = specific or supported
    main: str | None = choices[0] if choices else None
    if level == "low":
        lines.append(
            "Verdict: low risk - nothing in today's activity stands out enough to alert."
        )
        lines.append("What to do: no action needed.")
    elif main is None:
        lines.append(
            f"Verdict: {level} risk, but the features supplied do not show which "
            "activity drove it."
        )
        lines.append("What to do: pull the user's full activity log for the day.")
    else:
        threat, todo = _VERDICTS[main]
        lines.append(f"Verdict: {level} risk of {threat}.")
        lines.append(f"What to do: {todo}.")
    return [f"{i}. {t}" for i, t in enumerate(lines, 1)]


def score_user_day(
    features: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Score one user-day. Missing features get the training median."""
    b = bundle if bundle is not None else load_bundle(root)
    cols = b["features"]
    row = []
    for c in cols:
        v = _num(features, c)
        row.append(b["fill"][c] if v is None or not np.isfinite(v) else v)
    x = np.asarray([row], dtype=np.float32)
    risk = float(model_scores(b["family"], b["model"], x)[0])
    return {
        "risk": round(risk, 4),
        "verdict": _verdict_word(risk, b),
        "threshold": round(float(b["threshold"]), 4),
        "model": b["family"],
        "why": explain(b, features, risk, x),
    }


# -------------------------------------------------------------- SOC: improve


def improve(
    new_labelled_data: pd.DataFrame, root: Path | None = None
) -> dict[str, Any]:
    """Champion/challenger update with new labelled user-days.

    ``new_labelled_data`` holds the feature columns plus ``label`` (1 =
    malicious) and optionally ``split`` ('train' or 'val'; default 'train').
    The challenger (same family and parameters, refit on old + new train) is
    kept only if its PR-AUC on the validation set (old + new val) is not worse
    than the champion's; the champion then moves to ``prev/``.
    """
    champ = load_bundle(root)
    cols, fam = champ["features"], champ["family"]
    old_tr, old_va = _load_store_data(root)
    new = new_labelled_data.copy()
    for c in cols:
        if c not in new.columns:
            new[c] = champ["fill"][c]
    split = (
        new["split"] if "split" in new.columns else pd.Series("train", index=new.index)
    )
    keep = [*cols, "label"]
    tr = pd.concat([old_tr[keep], new.loc[split != "val", keep]], ignore_index=True)
    va = pd.concat([old_va[keep], new.loc[split == "val", keep]], ignore_index=True)
    X_tr, y_tr = tr[cols].to_numpy(np.float32), tr["label"].to_numpy()
    X_va, y_va = va[cols].to_numpy(np.float32), va["label"].to_numpy()
    model = make_model(fam, 0, champ["params"])
    if fam == "isoforest":
        model.fit(X_tr)
    else:
        model.fit(X_tr, y_tr)
    challenger = make_bundle(fam, model, champ["params"], cols, X_va, y_va, X_tr)
    champ_ap = _pr_auc(y_va, model_scores(fam, champ["model"], X_va))
    chall_ap = float(challenger["val_pr_auc"])
    champ_speed = items_per_second(
        lambda: model_scores(fam, champ["model"], X_va), len(X_va)
    )
    promoted = bool(np.nan_to_num(chall_ap, nan=-1) >= np.nan_to_num(champ_ap, nan=-1))
    if promoted:
        save_bundle(challenger, root)
        save_store_data(tr, va, root)
    return {
        "decision": "promote challenger" if promoted else "keep champion",
        "promoted": promoted,
        "metric": "val_pr_auc",
        "champion": round(champ_ap, 4),
        "challenger": round(chall_ap, 4),
        "champion_rows_per_s": champ_speed,
        "challenger_rows_per_s": challenger["rows_per_s"],
        "new_rows": int(len(new)),
        "prev_kept_at": str(area_dir(root) / "prev") if promoted else None,
    }


# ----------------------------------------------------------- SOC: blind test


def blind_test(
    release_dir: str | Path,
    answers_dir: str | Path | None = None,
    release: str = "4.1",
    root: Path | None = None,
) -> dict[str, Any]:
    """Score an unseen CERT release with the FROZEN model and threshold.

    No refit and no re-thresholding. With ``answers_dir`` the user-day and
    user-level detection metrics are returned; otherwise only the alert load.
    """
    b = load_bundle(root)
    df = feats.build_features(Path(release_dir), cache_dir=None)
    X = df[b["features"]].to_numpy(np.float32)
    t0 = time.perf_counter()
    s = model_scores(b["family"], b["model"], X)
    secs = time.perf_counter() - t0
    out: dict[str, Any] = {
        "release": release,
        "model": b["family"],
        "threshold": b["threshold"],
        "rows": int(len(df)),
        "days": int(df["day"].nunique()),
        "alerts": int((s >= b["threshold"]).sum()),
        "rows_per_s": round(len(df) / max(secs, 1e-9), 1),
    }
    out["alerts_per_day"] = round(out["alerts"] / max(out["days"], 1), 2)
    if answers_dir is not None:
        mal = feats.malicious_days(Path(answers_dir), release)
        df = feats.attach_labels(df, mal)
        y = df["label"].to_numpy()
        out["metrics"] = point_metrics(y, s, b["threshold"], df["day"].nunique())
        out["metrics"]["recall_at_budget"] = recall_at_budget(df, s, BUDGET_K)
        users = user_level(df, s, b["threshold"])
        out["users"] = users
        out["users_detected"] = f"{sum(u['detected'] for u in users)}/{len(users)}"
    return out
