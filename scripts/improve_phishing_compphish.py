"""Phishing page model with 2024-25 pages (owner 2026-10-25) - NOT BLIND.

CompPhish (2024-09 .. 2025-08) was scored once in blind round 2 (page model recall 0.44:
the 2021 training pages do not carry over), then released by the owner for development.
Split by registered domain: 70 % joins the training data, 30 % is the check. The
candidate (Mendeley 2021 train + CompPhish 70 %) replaces the deployed page model only if
precision AND recall do not drop on the old held-out set (Mendeley 2021 test), after a
backup; the CompPhish check is reported.

    python -m scripts.improve_phishing_compphish [--no-save]
Writes results/extra/improve_phishing_compphish.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts import train_phishing_page as tpp
from utils.paths import PROJECT_ROOT, RESULTS_DIR

CP = Path("D:/SplitShield-data/blind2/compphish")
CACHE = Path("D:/SplitShield-data/cache/phish_page/compphish.npz")


def compphish() -> dict[str, np.ndarray]:
    if CACHE.is_file():
        return dict(np.load(CACHE))
    from scripts.blind_test_v2 import _xlsx_rows
    from src.mlareas.phishing import registered_domain, split_url
    from src.mlareas.phishing_page import page_features

    rows = _xlsx_rows(CP / "Mapping_File.xlsx")[1:]
    X, y, doms = [], [], []
    with zipfile.ZipFile(CP / "All_HTML.zip") as z:
        files = {Path(m).stem: m for m in z.namelist() if not m.endswith("/")}
        for r in rows:
            if len(r) < 3 or r[2] not in ("0", "1") or str(r[0]).split(".")[0] not in files:
                continue
            html = z.read(files[str(r[0]).split(".")[0]]).decode("utf-8", "replace")
            X.append(page_features(html, r[1]))
            y.append(int(r[2]))
            doms.append(registered_domain(split_url(r[1])[0]))
    out: dict[str, np.ndarray] = {"X": np.vstack(X).astype(np.float32), "y": np.array(y, np.int8),
                                  "domain": np.array(doms).astype(str)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def main() -> None:
    import joblib
    import lightgbm
    from lightgbm import LGBMClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--owner-exception", default="", help="deploy although the rule fails, with this reason")
    a = ap.parse_args()
    deployed_p = PROJECT_ROOT / "artifacts" / "mlareas" / "phishing" / "page" / "page_model.joblib"
    champ = joblib.load(deployed_p)
    keep = champ["keep"]
    m = tpp.build(tpp.ZIP, Path("D:/SplitShield-data/cache/phish_page"))
    mtr, mva, mte = tpp.group_split(m["domain"])
    c = compphish()
    u = np.unique(c["domain"])
    rng = np.random.default_rng(0)
    rng.shuffle(u)
    check_d = set(u[: int(0.3 * len(u))])
    is_check = np.array([d in check_d for d in c["domain"]])
    ctrain = np.flatnonzero(~is_check)
    rng.shuffle(ctrain)
    cval, ctr = ctrain[: int(0.15 * len(ctrain))], ctrain[int(0.15 * len(ctrain)):]
    Xtr = np.vstack([m["X"][mtr][:, keep], c["X"][ctr][:, keep]])
    ytr = np.concatenate([m["y"][mtr], c["y"][ctr]]).astype(int)
    Xva = np.vstack([m["X"][mva][:, keep], c["X"][cval][:, keep]])
    yva = np.concatenate([m["y"][mva], c["y"][cval]]).astype(int)
    cand = LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=63, min_child_samples=30, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0, random_state=0, n_jobs=8, verbose=-1)  # fmt: skip
    cand.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
             callbacks=[lightgbm.early_stopping(60, verbose=False)])  # fmt: skip
    pv = cand.predict_proba(Xva)[:, 1]
    neg = np.sort(pv[yva == 0])[::-1]
    thr = float(max(0.5, np.nextafter(neg[int(0.01 * len(neg))], 1.0)))

    def rates(model: Any, t: float, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return tpp.rates(model.predict_proba(X)[:, 1], y.astype(int), t)

    old_test = (m["X"][mte][:, keep], m["y"][mte])
    chk = (c["X"][is_check][:, keep], c["y"][is_check])
    res: dict[str, Any] = {
        "label": "NOT BLIND: CompPhish scored once in blind round 2, then released for development",
        "split": "CompPhish by registered domain: 70 % train (15 % of it validation), 30 % check",
        "champion": {"mendeley_test": rates(champ["model"], champ["threshold"], *old_test),
                     "compphish_check": rates(champ["model"], champ["threshold"], *chk)},
        "candidate": {"threshold": thr, "mendeley_test": rates(cand, thr, *old_test),
                      "compphish_check": rates(cand, thr, *chk)}}  # fmt: skip
    o, n = res["champion"]["mendeley_test"], res["candidate"]["mendeley_test"]
    ok = n["precision"] >= o["precision"] - 1e-12 and n["recall"] >= o["recall"] - 1e-12
    res["decision"] = "promoted" if ok else "kept champion (old held-out precision or recall would drop)"
    if not ok and a.owner_exception:
        res["decision"], res["exception"] = "promoted as an owner exception", a.owner_exception
        ok = True
    if ok and not a.no_save:
        bak = PROJECT_ROOT / "artifacts" / "deploy_backup" / f"{time.strftime('%Y%m%d-%H%M')}_pre_page_model_v2"
        bak.mkdir(parents=True, exist_ok=True)
        shutil.copy2(deployed_p, bak / deployed_p.name)
        joblib.dump(champ | {"model": cand, "threshold": thr, "trained_on": "Mendeley 2021 + CompPhish 70 %",
                             "trained": time.strftime("%Y-%m-%d %H:%M"), "metrics": res["candidate"]},
                    deployed_p, compress=3)  # fmt: skip
        res["backup"] = str(bak)
    (RESULTS_DIR / "extra" / "improve_phishing_compphish.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
