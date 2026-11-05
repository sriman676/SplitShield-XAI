"""Improve the phishing model's recall with real legitimate domains (owner 2026-10-15).

Problem: PhiUSIIL's legitimate hosts do not look like real popular sites, so the
deployed model (v3) needs a high threshold (0.955) to keep false alarms on real
sites low, and recall on PhiUSIIL test phishing drops to 0.590.

Fix tried here: add real legitimate domains to TRAINING, from Tranco ranks
200,001-1,000,000 - disjoint from every other Tranco range in use:
  ranks       1-20,000    blind pool (OpenPhish + Tranco blind test): never used
  ranks  20,001-100,000   validation sample (threshold choice; as phishing_threshold.py)
  ranks 100,001-200,000   test sample (reported only)
  ranks 200,001-1,000,000 NEW training legitimate domains (this script)
Half of them get a "www." prefix (PhiUSIIL hosts usually have one).

Candidates (LightGBM settings, number of extra domains) are fitted on PhiUSIIL train
hosts + the extra domains, calibrated (isotonic) on validation, and given a
threshold on validation by the same rule as v3 (max validation F1 with <= FPR_CAP of
the Tranco validation sample flagged). The best candidate on validation is compared
with the deployed model on TEST: it replaces it only if PhiUSIIL-test precision AND
recall are both not worse AND <= 2 % of the Tranco test sample is flagged.

    python -m scripts.improve_phishing [--no-save]
    python -m scripts.improve_phishing --mode host2 [--no-save]

``--mode host2`` (owner 2026-10-16, underfitting: train recall ~= test recall): the
same procedure with richer HOSTNAME features (brand edit distance, TLD statistics,
2-/3-/4-gram hashes, bigram rarity, ...; src/mlareas/phishing.py HOST2_EXTRA). The
leaking full-URL / all-dataset columns are never used. Results go to
results/extra/improve_phishing_host2.json.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.request
import zipfile
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts import phishing_threshold as pt
from src.mlareas import common, phishing
from utils.paths import PROJECT_ROOT, RESULTS_DIR

TRAIN_RANKS = (200_001, 1_000_000)
FPR_CAP = pt.FPR_CAP
TEST_FPR_MAX = 0.02
CACHE = PROJECT_ROOT / "data" / "raw" / "tranco" / "top-1m.csv"
OUT = RESULTS_DIR / "extra"
CONFIGS_HOST2 = [
    {"n_extra": 150_000, "n_estimators": 800, "num_leaves": 127},
    {"n_extra": 300_000, "n_estimators": 800, "num_leaves": 127},
    {"n_extra": 300_000, "n_estimators": 1200, "num_leaves": 255},
]
CONFIGS = [
    {"n_extra": 50_000, "n_estimators": 400, "num_leaves": 63},
    {"n_extra": 150_000, "n_estimators": 400, "num_leaves": 63},
    {"n_extra": 150_000, "n_estimators": 800, "num_leaves": 127},
    {"n_extra": 300_000, "n_estimators": 800, "num_leaves": 127},
]


def tranco() -> pd.DataFrame:
    if not CACHE.is_file():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request("https://tranco-list.eu/top-1m.csv.zip",
                                     headers={"User-Agent": "splitshield-improve"})  # fmt: skip
        raw = urllib.request.urlopen(req, timeout=300).read()  # noqa: S310 - fixed https URL
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            CACHE.write_bytes(z.read("top-1m.csv"))
    return pd.read_csv(CACHE, header=None, names=["rank", "domain"])


def extra_hosts(
    df: pd.DataFrame, n: int, exclude: set[str], seed: int = 7
) -> list[str]:
    pool = df[(df["rank"] >= TRAIN_RANKS[0]) & (df["rank"] <= TRAIN_RANKS[1])]["domain"]
    doms = [d for d in pool.astype(str).str.lower().tolist() if d not in exclude]
    random.Random(seed).shuffle(doms)
    out = []
    for i, d in enumerate(doms[:n]):
        out.append(f"www.{d}" if i % 2 == 0 else d)
    return out


def make(cfg: dict[str, Any]) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=cfg["n_estimators"], learning_rate=0.05, num_leaves=cfg["num_leaves"],
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
        n_jobs=6, verbose=-1,
    )  # fmt: skip


def evaluate(model: Any, iso: Any, thr: float, sets: dict[str, Any]) -> dict[str, Any]:
    def p(X: np.ndarray) -> np.ndarray:
        raw = phishing.predict_proba(model, X)
        return np.asarray(iso.predict(raw) if iso is not None else raw, np.float64)

    pv, ptv, pte, ptt = p(sets["Xv"]), p(sets["Xtv"]), p(sets["Xte"]), p(sets["Xtt"])
    return {
        "threshold": thr,
        "validation_phiusiil": phishing.binary_metrics(sets["yv"], pv, thr),
        "validation_tranco_fpr": float((ptv >= thr).mean()),
        "test_phiusiil": phishing.binary_metrics(sets["yte"], pte, thr),
        "test_tranco_100k_200k_fpr": float((ptt >= thr).mean()),
        "_p": (pv, ptv),
    }


def choose_keep_precision(
    p_val: np.ndarray, y_val: np.ndarray, p_tr: np.ndarray, min_prec: float
) -> tuple[float, bool]:
    """Validation rule: the threshold with the highest PhiUSIIL-validation RECALL whose
    validation precision is >= ``min_prec`` (the deployed model's) and that flags
    <= FPR_CAP of the Tranco validation sample. (False, highest threshold) if none."""
    best: tuple[float, float] | None = None
    for t in np.unique(np.concatenate([p_val, p_tr])):
        if t <= phishing.SUSPICIOUS_AT:
            continue
        m = phishing.binary_metrics(y_val, p_val, float(t))
        if m["precision"] >= min_prec and float((p_tr >= t).mean()) <= FPR_CAP:
            if best is None or m["recall"] > best[1]:
                best = (float(t), m["recall"])
    if best is None:
        return float(np.max(p_val)), False
    return best[0], True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--mode", default="host", choices=("host", "host2"))
    a = ap.parse_args()
    mode = a.mode
    t0 = time.time()
    champ = phishing.load(refresh=True)
    data = phishing.load_phiusiil()
    X, y, _names, keys, _stats = phishing._variant_data(data, "url_host")
    tr, va, te = phishing.split_indices(y, int(champ.get("seed", 0)))
    if [keys[i] for i in va] != list(champ["val_hosts"]):
        raise RuntimeError("rebuilt validation split differs from the deployed bundle")
    tdf = tranco()
    t_val, t_test = pt._sample(tdf, pt.VAL_RANKS, 0), pt._sample(tdf, pt.TEST_RANKS, 1)
    used = {phishing.split_url(u)[0].removeprefix("www.") for u in t_val + t_test}
    used |= {k.removeprefix("www.") for k in keys}  # never a PhiUSIIL host either
    sets = {
        "Xv": X[va], "yv": y[va], "Xte": X[te], "yte": y[te],
        "Xtv": phishing.featurize(t_val, "host"), "Xtt": phishing.featurize(t_test, "host"),
    }  # fmt: skip
    champ_sets = sets  # the champion is always scored on its own (host) features
    if mode != "host":
        X = phishing.featurize(keys, mode)
        sets = {
            "Xv": X[va], "yv": y[va], "Xte": X[te], "yte": y[te],
            "Xtv": phishing.featurize(t_val, mode), "Xtt": phishing.featurize(t_test, mode),
        }  # fmt: skip

    # champion, measured in the same harness
    c = evaluate(
        champ["model"], champ.get("calibrator"), float(champ["threshold"]), champ_sets
    )
    min_prec = float(c["validation_phiusiil"]["precision"])
    rows: list[dict[str, Any]] = []
    extra_cache: dict[int, np.ndarray] = {}
    for cfg in CONFIGS if mode == "host" else CONFIGS_HOST2:
        n = cfg["n_extra"]
        if n not in extra_cache:
            extra_cache[n] = phishing.featurize(extra_hosts(tdf, n, used), mode)
        Xtr = np.vstack([X[tr], extra_cache[n]])
        ytr = np.concatenate([y[tr], np.zeros(n, np.int8)])
        model = make(cfg).fit(Xtr, ytr)
        raw_v = phishing.predict_proba(model, sets["Xv"])
        raw_tv = phishing.predict_proba(model, sets["Xtv"])
        # calibrate on the whole validation set (PhiUSIIL + real popular sites)
        iso = phishing.calibrate(np.concatenate([raw_v, raw_tv]),
                                 np.concatenate([sets["yv"], np.zeros(len(raw_tv), int)]))  # fmt: skip
        pv, ptv = np.asarray(iso.predict(raw_v)), np.asarray(iso.predict(raw_tv))
        thr, met = choose_keep_precision(pv, sets["yv"], ptv, min_prec)
        r = evaluate(model, iso, thr, sets)
        r |= {
            "config": cfg,
            "constraint_met": met,
            "_model": model,
            "_iso": iso,
        }
        rows.append(r)
        print(cfg, "val P/R", round(r["validation_phiusiil"]["precision"], 4),
              round(r["validation_phiusiil"]["recall"], 4),
              "val tranco fpr", round(r["validation_tranco_fpr"], 4), flush=True)  # fmt: skip
    ok = [r for r in rows if r["constraint_met"]]
    best = max(ok or rows, key=lambda r: r["validation_phiusiil"]["recall"])
    bt, ct = best["test_phiusiil"], c["test_phiusiil"]
    keep = (bt["precision"] >= ct["precision"] - 1e-12 and bt["recall"] >= ct["recall"] - 1e-12
            and best["test_tranco_100k_200k_fpr"] <= TEST_FPR_MAX)  # fmt: skip
    clean = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}  # noqa: E731
    doc: dict[str, Any] = {
        "what": "phishing model retrained with real legitimate domains (Tranco ranks "
                f"{TRAIN_RANKS[0]:,}-{TRAIN_RANKS[1]:,}); blind pool ranks 1-20,000 never used; "
                f"features: {mode}",
        "mode": mode,
        "tranco_ranges": {"blind_never_used": [1, 20_000], "validation": pt.VAL_RANKS,
                          "test_report_only": pt.TEST_RANKS, "training_extra": TRAIN_RANKS},
        "rule": f"choose on validation (max recall with precision >= deployed and Tranco-validation FPR <= {FPR_CAP:.1%}); "
                "replace only if test precision AND recall not worse and Tranco-test FPR "
                f"<= {TEST_FPR_MAX:.0%}",
        "champion": clean(c), "candidates": [clean(r) for r in rows], "best": clean(best),
        "decision": "promoted" if keep else "kept champion", "saved": False,
        "seconds": round(time.time() - t0),
    }  # fmt: skip
    if keep and not a.no_save:
        cand = {k: v for k, v in champ.items() if k != "_det"}
        cand.update(mode=mode, feature_names=phishing.feature_names(mode),
                    model=best["_model"], calibrator=best["_iso"], threshold=best["threshold"],
                    version=int(champ.get("version", 1)) + 1, metrics=bt,
                    threshold_basis=doc["rule"], extra_legit_domains=best["config"]["n_extra"],
                    suspicious_at=float(min(best["threshold"], float(champ.get("suspicious_at", 0.5)))))  # fmt: skip
        # the 'suspicious' band: lowest probability reached by <= 5 % of Tranco validation
        ptv = best["_p"][1]
        cands = sorted(float(x) for x in np.unique(ptv) if x <= best["threshold"])
        cand["suspicious_at"] = next((s for s in cands if (ptv >= s).mean() <= pt.SUSPICIOUS_CAP),
                                     best["threshold"])  # fmt: skip
        common.save_bundle(phishing.AREA, cand)
        phishing.load(refresh=True)
        doc["saved"], doc["version"] = True, cand["version"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("improve_phishing.json" if mode == "host" else f"improve_phishing_{mode}.json")).write_text(
        json.dumps(doc, indent=1, default=float), "utf-8"
    )
    print(json.dumps({"champion_test": ct, "best_test": bt,
                      "best_tranco_test_fpr": best["test_tranco_100k_200k_fpr"],
                      "decision": doc["decision"], "saved": doc["saved"]}, indent=1, default=float))  # fmt: skip


if __name__ == "__main__":
    main()
