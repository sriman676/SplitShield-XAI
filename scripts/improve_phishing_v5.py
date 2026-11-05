"""Phishing URL model v5 (owner 2026-10-24): more phishing hosts in training.

The deployed host model (v4: PhiUSIIL + 300,000 real legitimate domains, ``host2``
features) misses phishing on ordinary-looking hosts (recall 0.59 on PhiUSIIL test,
0.74 on the 2026-10-24 blind round). v5 adds the hosts of the Mendeley 2021 phishing
set (PhishTank / OpenPhish / PhishRepo phishing and Google-result legitimate sites) -
TRAINING domains only (the same domain split as scripts/train_phishing_page.py) - to
v4's training data. Everything else is v4's procedure (scripts/improve_phishing.py):
calibration and threshold on validation, and the owner rule on TEST: replace only if
PhiUSIIL-test precision AND recall are not worse and <= 2 % of the Tranco test sample
(ranks 100,001-200,000) is flagged. The new blind set (CompPhish) is never read here.

    python -m scripts.improve_phishing_v5 [--no-save] [--cache DIR]
Writes results/extra/improve_phishing_v5.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts import improve_phishing as ip
from scripts import phishing_threshold as pt
from scripts import train_phishing_page as tpp
from src.mlareas import common, phishing
from utils.paths import RESULTS_DIR

MODE = "host2"
CFG = {"n_extra": 300_000, "n_estimators": 800, "num_leaves": 127}  # v4's chosen setting


def mendeley_hosts(cache: Path, exclude: set[str]) -> dict[str, Any]:
    d = tpp.build(tpp.ZIP, cache)
    tr, _va, te = tpp.group_split(d["domain"])
    def hosts(idx: np.ndarray) -> tuple[list[str], np.ndarray]:
        seen: dict[str, int] = {}
        for i in idx:
            h = phishing.split_url(str(d["url"][i]))[0].lower()
            if h and h.removeprefix("www.") not in exclude:
                seen[h] = max(seen.get(h, 0), int(d["y"][i]))
        return list(seen), np.array(list(seen.values()), np.int8)

    htr, ytr = hosts(tr)
    hte, yte = hosts(te)
    return {"train_hosts": htr, "y_train": ytr, "test_hosts": hte, "y_test": yte}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path("D:/SplitShield-data/cache/phish_page"))
    a = ap.parse_args()
    t0 = time.time()
    champ = phishing.load(refresh=True)
    data = phishing.load_phiusiil()
    _X, y, _names, keys, _stats = phishing._variant_data(data, "url_host")
    tr, va, te = phishing.split_indices(y, int(champ.get("seed", 0)))
    if [keys[i] for i in va] != list(champ["val_hosts"]):
        raise RuntimeError("rebuilt validation split differs from the deployed bundle")
    tdf = ip.tranco()
    t_val, t_test = pt._sample(tdf, pt.VAL_RANKS, 0), pt._sample(tdf, pt.TEST_RANKS, 1)
    used = {phishing.split_url(u)[0].removeprefix("www.") for u in t_val + t_test}
    used |= {keys[i].removeprefix("www.") for i in (*va, *te)}  # PhiUSIIL val / test hosts
    X = phishing.featurize(keys, MODE)
    sets = {"Xv": X[va], "yv": y[va], "Xte": X[te], "yte": y[te],
            "Xtv": phishing.featurize(t_val, MODE), "Xtt": phishing.featurize(t_test, MODE)}  # fmt: skip
    champ_mode = str(champ.get("mode", "host"))
    cX = phishing.featurize(keys, champ_mode)
    champ_sets = {"Xv": cX[va], "yv": y[va], "Xte": cX[te], "yte": y[te],
                  "Xtv": phishing.featurize(t_val, champ_mode), "Xtt": phishing.featurize(t_test, champ_mode)}  # fmt: skip
    c = ip.evaluate(champ["model"], champ.get("calibrator"), float(champ["threshold"]), champ_sets)
    min_prec = float(c["validation_phiusiil"]["precision"])
    m = mendeley_hosts(a.cache, used)
    extra = phishing.featurize(ip.extra_hosts(tdf, CFG["n_extra"], used), MODE)
    Xm, Xmt = phishing.featurize(m["train_hosts"], MODE), phishing.featurize(m["test_hosts"], MODE)
    Xtr = np.vstack([X[tr], extra, Xm])
    ytr = np.concatenate([y[tr], np.zeros(len(extra), np.int8), m["y_train"]])
    model = ip.make(CFG).fit(Xtr, ytr)
    raw_v, raw_tv = phishing.predict_proba(model, sets["Xv"]), phishing.predict_proba(model, sets["Xtv"])
    iso = phishing.calibrate(np.concatenate([raw_v, raw_tv]), np.concatenate([sets["yv"], np.zeros(len(raw_tv), int)]))
    pv, ptv = np.asarray(iso.predict(raw_v)), np.asarray(iso.predict(raw_tv))
    thr, met = ip.choose_keep_precision(pv, sets["yv"], ptv, min_prec)
    r = ip.evaluate(model, iso, thr, sets)
    pm_new = np.asarray(iso.predict(phishing.predict_proba(model, Xmt)))
    cm = phishing.featurize(m["test_hosts"], champ_mode)
    raw_c = phishing.predict_proba(champ["model"], cm)
    pm_old = np.asarray(champ["calibrator"].predict(raw_c)) if champ.get("calibrator") is not None else raw_c
    bt, ct = r["test_phiusiil"], c["test_phiusiil"]
    keep = (met and bt["precision"] >= ct["precision"] - 1e-12 and bt["recall"] >= ct["recall"] - 1e-12
            and r["test_tranco_100k_200k_fpr"] <= ip.TEST_FPR_MAX)  # fmt: skip
    clean = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    doc: dict[str, Any] = {
        "what": "phishing URL model v5: v4 training data + Mendeley 2021 training-domain hosts",
        "mendeley_hosts": {"train": len(m["train_hosts"]), "train_phishing": int(m["y_train"].sum()),
                           "test": len(m["test_hosts"]), "test_phishing": int(m["y_test"].sum())},
        "champion": clean(c), "candidate": clean(r), "constraint_met": met,
        "mendeley_test": {"champion": phishing.binary_metrics(m["y_test"], pm_old, float(champ["threshold"])),
                          "candidate": phishing.binary_metrics(m["y_test"], pm_new, thr)},
        "rule": "replace only if PhiUSIIL-test precision AND recall not worse and Tranco-test FPR <= 2 %",
        "decision": "promoted" if keep else "kept champion", "saved": False, "blind_used": False,
        "seconds": round(time.time() - t0)}  # fmt: skip
    if keep and not a.no_save:
        cand = {k: v for k, v in champ.items() if k != "_det"}
        cand.update(mode=MODE, feature_names=phishing.feature_names(MODE), model=model, calibrator=iso,
                    threshold=thr, version=int(champ.get("version", 1)) + 1, metrics=bt,
                    threshold_basis=doc["rule"], extra_phishing_hosts="Mendeley 2021 (train domains)")  # fmt: skip
        cands = sorted(float(x) for x in np.unique(ptv) if x <= thr)
        cand["suspicious_at"] = next((s for s in cands if (ptv >= s).mean() <= pt.SUSPICIOUS_CAP), thr)
        common.save_bundle(phishing.AREA, cand)
        doc["saved"], doc["version"] = True, cand["version"]
    RESULTS_DIR.joinpath("extra").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "extra" / "improve_phishing_v5.json").write_text(json.dumps(doc, indent=1, default=float), "utf-8")
    print(json.dumps({"champion_test": ct, "candidate_test": bt, "tranco_test_fpr": r["test_tranco_100k_200k_fpr"],
                      "mendeley_test": doc["mendeley_test"], "decision": doc["decision"]}, indent=1, default=float))  # fmt: skip


if __name__ == "__main__":
    main()
