"""Prune the tree models (owner 2026-10-16): smaller / faster with the same answers.

    python -m scripts.prune_models zeek     [--no-save]
    python -m scripts.prune_models malware  [--no-save]

For each model, smaller variants are tried (fewer trees; for the forest also a depth
limit). On VALIDATION the smallest variant whose precision AND recall are both not
below the deployed model's is picked; it replaces the deployed model only if on TEST
precision AND recall are also not worse and it is smaller (owner rule: an
improvement is kept only if held-out precision and recall do not drop). The deployed
model is backed up first. Blind sets are never read.
Results: results/extra/prune_<model>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import time
from typing import Any

import joblib
import numpy as np

from utils.paths import PROJECT_ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "extra"


def _pr(p: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float]:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "false_alarm_rate": fp / max(1, int((y == 0).sum()))}  # fmt: skip


def _not_worse(a: dict[str, float], b: dict[str, float]) -> bool:
    return a["precision"] >= b["precision"] and a["recall"] >= b["recall"]


def _speed(fn: Any, X: np.ndarray) -> float:
    t = time.perf_counter()
    fn(X)
    return len(X) / max(time.perf_counter() - t, 1e-9)


# --- Zeek random forest ------------------------------------------------------------------


def prune_zeek(save: bool) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    from scripts.improve_zeek import _data
    from src.soc import zeekml

    data = _data()
    bundle = joblib.load(zeekml.MODEL)
    champ, thr = bundle["model"], float(bundle["threshold"])
    Xv, yv = data["val"]
    Xt, yt = data["test"]
    base_v = _pr(champ.predict_proba(Xv)[:, 1], yv, thr)
    base_t = _pr(champ.predict_proba(Xt)[:, 1], yt, thr)
    size0 = len(pickle.dumps(champ))
    cands: list[tuple[str, Any]] = []
    for k in (25, 50, 75, 100, 150):  # the first k trees of the deployed forest
        m = copy.copy(champ)
        m.estimators_ = champ.estimators_[:k]
        m.n_estimators = k
        cands.append((f"first {k} trees", m))
    params = champ.get_params()
    for depth in (20, 30):  # same forest settings, depth-limited (retrained)
        m = RandomForestClassifier(**(params | {"max_depth": depth, "n_jobs": -1}))
        cands.append((f"retrained, max depth {depth}", m.fit(*data["train"])))
    rows: list[dict[str, Any]] = []
    for name, m in cands:
        v = _pr(m.predict_proba(Xv)[:, 1], yv, thr)
        rows.append({"variant": name, "val": v, "size_mb": round(len(pickle.dumps(m)) / 2**20, 1),
                     "ok_on_validation": _not_worse(v, base_v), "_m": m})  # fmt: skip
    ok = [r for r in rows if r["ok_on_validation"]]
    doc: dict[str, Any] = {
        "model": "zeek", "rule": "smallest variant not worse on validation; kept only if test "
        "precision AND recall not worse and smaller", "blind_used": False,
        "deployed": {"val": base_v, "test": base_t, "size_mb": round(size0 / 2**20, 1),
                     "conns_per_s": round(_speed(champ.predict_proba, Xt[:50_000]))},
    }  # fmt: skip
    decision = "kept: no smaller variant was as good on validation"
    if ok:
        pick = min(ok, key=lambda r: r["size_mb"])
        m = pick["_m"]
        t = _pr(m.predict_proba(Xt)[:, 1], yt, thr)
        pick["test"] = t
        pick["conns_per_s"] = round(_speed(m.predict_proba, Xt[:50_000]))
        doc["picked"] = {k: v for k, v in pick.items() if k != "_m"}
        if _not_worse(t, base_t) and pick["size_mb"] < doc["deployed"]["size_mb"]:
            decision = f"pruned: {pick['variant']}"
            if save:
                backup = zeekml.MODEL.with_name(
                    f"model_before_prune_{time.strftime('%Y%m%d-%H%M')}.joblib"
                )
                shutil.copy2(zeekml.MODEL, backup)
                joblib.dump(
                    bundle | {"model": m, "pruned": pick["variant"]}, zeekml.MODEL
                )
                doc["backup"] = str(backup.relative_to(PROJECT_ROOT))
        else:
            decision = "kept: the validation pick was worse on test"
    doc["variants"] = [{k: v for k, v in r.items() if k != "_m"} for r in rows]
    doc["decision"] = decision
    return doc


# --- malware LightGBM ----------------------------------------------------------------------


def prune_malware(save: bool) -> dict[str, Any]:
    """Fewer boosting rounds: the same trees, cut after k iterations. Judged as the
    deployed model is: detection and precision at the 1 %-FPR threshold set on
    validation (scripts/improve_malware.py); a pruned model gets its own thresholds,
    again from validation."""
    from scripts import train_malware as tm
    from scripts.improve_malware import at_threshold
    from src.mlareas import malware
    from src.mlareas.malware_data import Stacked, free_ram_gb

    if free_ram_gb() < 6:
        raise MemoryError(f"only {free_ram_gb():.1f} GB free; need 6 GB")
    d = tm.load_split()
    xs, xt = Stacked(d["tr_paths"]), Stacked(d["te_paths"])
    x_val, y_val = xs.take(d["split"].val), d["mtr"]["label"][d["split"].val]
    test_rows = d["test_rows"]
    y_test = d["mte"]["label"][test_rows]
    champ, cmeta = malware.load_champion()
    total = champ.num_trees()
    c_thr = float(cmeta["thresholds"]["fpr_0.01"])
    base_v = at_threshold(np.asarray(champ.predict(x_val)), y_val, c_thr)
    base_t = at_threshold(
        tm.predict_chunks(champ.predict, xt, test_rows), y_test, c_thr
    )

    def ok(a: dict[str, float], b: dict[str, float]) -> bool:
        return a["detection"] >= b["detection"] and a["precision"] >= b["precision"]

    rows: list[dict[str, Any]] = []
    for frac in (0.25, 0.4, 0.5, 0.6, 0.75):
        k = max(1, int(total * frac))
        sv = np.asarray(champ.predict(x_val, num_iteration=k))
        thr = malware.threshold_at_fpr(sv, y_val, 0.01)
        v = at_threshold(sv, y_val, thr)
        rows.append({"variant": f"first {k} of {total} trees", "k": k, "val": v,
                     "thresholds": {"fpr_0.01": thr, "fpr_0.001": malware.threshold_at_fpr(sv, y_val, 0.001)},
                     "ok_on_validation": ok(v, base_v)})  # fmt: skip
    doc: dict[str, Any] = {"model": "malware", "blind_used": False, "trees": total,
                           "rule": "fewest trees not worse on validation (detection AND precision at "
                                   "1 % FPR); kept only if test detection AND precision not worse",
                           "deployed": {"val": base_v, "test": base_t}, "variants": rows}  # fmt: skip
    decision = "kept: no shorter model was as good on validation"
    good = [r for r in rows if r["ok_on_validation"]]
    if good:
        pick = min(good, key=lambda r: r["k"])
        k = pick["k"]
        st = tm.predict_chunks(
            lambda x: champ.predict(x, num_iteration=k), xt, test_rows
        )
        pick["test"] = at_threshold(st, y_test, pick["thresholds"]["fpr_0.01"])
        doc["picked"] = pick
        if ok(pick["test"], base_t):
            decision = f"pruned: {pick['variant']}"
            if save:
                champ_dir = (
                    PROJECT_ROOT / "artifacts" / "mlareas" / "malware" / "champion"
                )
                backup = champ_dir.with_name(
                    f"champion_before_prune_{time.strftime('%Y%m%d-%H%M')}"
                )
                shutil.copytree(champ_dir, backup)
                champ.save_model(str(tm.WORK / "lgb_pruned.txt"), num_iteration=k)
                import lightgbm as lgb

                meta = dict(cmeta) | {"thresholds": pick["thresholds"], "test": pick["test"],
                                      "pruned": pick["variant"]}  # fmt: skip
                malware.save_champion(
                    lgb.Booster(model_file=str(tm.WORK / "lgb_pruned.txt")), meta
                )
                doc["backup"] = str(backup.relative_to(PROJECT_ROOT))
        else:
            decision = "kept: the validation pick was worse on test"
    doc["decision"] = decision
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=("zeek", "malware"))
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    doc = (prune_zeek if a.model == "zeek" else prune_malware)(not a.no_save)
    doc["seconds"] = round(time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"prune_{a.model}.json").write_text(
        json.dumps(doc, indent=1, default=str), "utf-8"
    )
    print(json.dumps({"decision": doc["decision"], "deployed": doc["deployed"],
                      "picked": doc.get("picked")}, indent=1, default=str))  # fmt: skip


if __name__ == "__main__":
    main()
