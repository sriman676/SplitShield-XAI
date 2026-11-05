"""Train the phishing PAGE model (owner 2026-10-24) on the Mendeley "Phishing Websites
Dataset" (2021, 80k URLs with their HTML, CC BY 4.0; training data only).

* Pages are read straight from the nested zips (nothing is unpacked to disk).
* Collection artefacts are stripped (src/mlareas/phishing_page.clean_html) and a leakage
  check is run: any single feature that separates the classes almost perfectly
  (ROC-AUC >= 0.98) is reported and dropped.
* Split by REGISTERED DOMAIN (no site in two splits), 70 / 15 / 15, seeded.
* LightGBM; threshold on VALIDATION = the lowest score at which at most 1 % of
  legitimate validation pages are flagged. TEST is reported once.

The new blind phishing set (CompPhish, D:/SplitShield-data/blind2/compphish) is never
read here.

    python -m scripts.train_phishing_page [--zip PATH] [--out DIR] [--cache DIR]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from utils.logging_config import configure_logging, get_logger
from utils.paths import PROJECT_ROOT, RESULTS_DIR

_log = get_logger(__name__)
ZIP = Path("D:/SplitShield-data/new/phish-html-2021/n96ncsr5g4-1.zip")
SEED = 0
FPR_BUDGET = 0.01
LEAK_AUC = 0.98
_ROW = re.compile(r"\((\d+),\s*'((?:[^'\\]|\\.)*)',\s*'([^']*)',\s*(\d)\s*,")


def index(z: zipfile.ZipFile) -> dict[str, tuple[str, int]]:
    sql = z.read("n96ncsr5g4-1/index.sql").decode("utf-8", errors="replace")
    return {m.group(3): (m.group(2).replace("\\'", "'"), int(m.group(4))) for m in _ROW.finditer(sql)}


def build(zpath: Path, cache: Path) -> dict[str, np.ndarray]:
    path = cache / "phish_page_2021.npz"
    if path.is_file():
        return dict(np.load(path, allow_pickle=False))
    from src.mlareas.phishing import registered_domain, split_url
    from src.mlareas.phishing_page import page_features

    z = zipfile.ZipFile(zpath)
    idx = index(z)
    X, y, urls, doms = [], [], [], []
    for part in sorted(n for n in z.namelist() if n.endswith(".zip")):
        inner = zipfile.ZipFile(io.BytesIO(z.read(part)))
        for name in inner.namelist():
            key = name.rsplit("/", 1)[-1]
            if key not in idx:
                continue
            url, lab = idx[key]
            html = inner.read(name).decode("utf-8", errors="replace")
            X.append(page_features(html, url))
            y.append(lab)
            urls.append(url)
            doms.append(registered_domain(split_url(url)[0]))
        _log.info("%s: %d pages so far", part.rsplit("/", 1)[-1], len(y))
    out: dict[str, np.ndarray] = {"X": np.vstack(X).astype(np.float32), "y": np.array(y, dtype=np.int8),
           "url": np.array(urls), "domain": np.array(doms)}  # fmt: skip
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def leakage(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    out = []
    for j, n in enumerate(names):
        col = X[:, j]
        if np.ptp(col) == 0:
            continue
        auc = roc_auc_score(y, col)
        auc = max(auc, 1 - auc)
        if auc >= LEAK_AUC:
            out.append({"feature": n, "auc": round(float(auc), 4)})
    return out


def group_split(domains: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.unique(domains)
    rng = np.random.default_rng(SEED)
    rng.shuffle(u)
    a, b = int(0.70 * len(u)), int(0.85 * len(u))
    tr, va = set(u[:a]), set(u[a:b])
    m_tr = np.array([d in tr for d in domains])
    m_va = np.array([d in va for d in domains])
    return np.flatnonzero(m_tr), np.flatnonzero(m_va), np.flatnonzero(~m_tr & ~m_va)


def rates(p: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float]:
    a = p >= thr
    tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    return {"precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, int((y == 1).sum())), 4),
            "false_alarm_rate": round(fp / max(1, int((y == 0).sum())), 4), "n": int(len(y))}  # fmt: skip


def run(zpath: Path, cache: Path, out: Path) -> dict[str, Any]:
    import joblib
    import lightgbm
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    from src.mlareas.phishing_page import feature_names

    d = build(zpath, cache)
    X, y = d["X"], d["y"].astype(int)
    names = feature_names()
    leaks = leakage(X, y, names)
    drop = {lk["feature"] for lk in leaks}
    keep = np.array([n not in drop for n in names])
    tr, va, te = group_split(d["domain"])
    t0 = time.time()
    m = LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=63, min_child_samples=30,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0,
                       random_state=SEED, n_jobs=8, verbose=-1)  # fmt: skip
    Xk = X[:, keep]
    m.fit(Xk[tr], y[tr], eval_set=[(Xk[va], y[va])], eval_metric="average_precision",
          callbacks=[lightgbm.early_stopping(60, verbose=False)])  # fmt: skip
    pv, pt, ptr = (m.predict_proba(Xk[i])[:, 1] for i in (va, te, tr))
    neg = np.sort(pv[y[va] == 0])[::-1]
    k = int(FPR_BUDGET * len(neg))
    thr = float(max(0.5, np.nextafter(neg[k], 1.0))) if len(neg) > k else 0.5
    res = {"what": "phishing page model (HTML + URL host), Mendeley 2021 set", "pages": int(len(y)),
           "phishing": int(y.sum()), "legitimate": int((y == 0).sum()),
           "split": "by registered domain 70/15/15", "leakage_dropped": leaks,
           "features_used": int(keep.sum()), "trees": int(m.best_iteration_ or 1500),
           "train_seconds": round(time.time() - t0), "threshold": thr,
           "auc": {"train": round(float(roc_auc_score(y[tr], ptr)), 4), "val": round(float(roc_auc_score(y[va], pv)), 4),
                   "test": round(float(roc_auc_score(y[te], pt)), 4)},
           "val": rates(pv, y[va], thr), "test": rates(pt, y[te], thr), "train": rates(ptr, y[tr], thr),
           "blind_used": False}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "threshold": thr, "keep": keep, "feature_names": [n for n, k in zip(names, keep, strict=True) if k],
                 "trained_on": "Mendeley n96ncsr5g4 v1 (2021)", "trained": time.strftime("%Y-%m-%d %H:%M"),
                 "metrics": res}, out / "page_model.joblib", compress=3)  # fmt: skip
    return res


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=ZIP)
    ap.add_argument("--cache", type=Path, default=Path("D:/SplitShield-data/cache/phish_page"))
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "deploy_candidates" / "phishing_page")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR / "extra")
    a = ap.parse_args()
    res = run(a.zip, a.cache, a.out)
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / "phishing_page.json").write_text(json.dumps(res, indent=1), "utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
