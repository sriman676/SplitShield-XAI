"""Improve the insider-threat model (CMU CERT r4.2, owner 2026-10-15).

Same data and time split as the deployed model (train 2010-01..2010-12, validation
2010-12..2011-03, test 2011-03..2011-05). A wider LightGBM grid (leaves, minimum
leaf size, learning rate, column sampling, class weighting) is searched with early
stopping on VALIDATION PR-AUC; the alert threshold is the validation-F1 maximum, as
for the deployed model. The best validation candidate replaces the deployed model only
if TEST precision AND recall are both not worse. The unseen-user split (seed 0) is
reported for the chosen setting. CERT r4.1 (blind set) is never used.

    python -m scripts.improve_insider [--no-save]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from typing import Any

import lightgbm as lgb
import numpy as np

from scripts import train_insider as ti
from src.mlareas import insider as ins
from utils.paths import RESULTS_DIR

OUT = RESULTS_DIR / "extra"
GRID = [
    {"num_leaves": nl, "min_child_samples": mc, "learning_rate": lr, "colsample_bytree": cs,
     "class_weight": cw}
    for nl, mc, lr, cs, cw in itertools.product((7, 15, 31), (10, 20, 50), (0.03, 0.06),
                                                (0.5, 0.8), ("balanced", None))
]  # fmt: skip


def fit(params: dict[str, Any], Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
        yva: np.ndarray, seed: int = 0) -> tuple[Any, dict[str, Any]]:  # fmt: skip
    m = ins.make_model("lightgbm", seed, {**params, "n_estimators": 2000})
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])  # fmt: skip
    return m, {**params, "n_estimators": int(m.best_iteration_ or 2000)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    df = ti.stage_features()
    champ = ins.load_bundle()
    cols = list(champ["features"])
    split = ins.time_split(df["day"])
    parts = {k: df[split == k] for k in ("train", "val", "test")}
    X = {k: v[cols].to_numpy(np.float32) for k, v in parts.items()}
    y = {k: v["label"].to_numpy() for k, v in parts.items()}
    n_days = {k: int(v["day"].nunique()) for k, v in parts.items()}

    def metrics(model: Any, thr: float, k: str) -> dict[str, Any]:
        s = ins.model_scores("lightgbm", model, X[k])
        return ins.point_metrics(y[k], s, thr, n_days[k])

    c_thr = float(champ["threshold"])
    champion = {k: metrics(champ["model"], c_thr, k) for k in ("val", "test")}
    print("champion", champion["test"], flush=True)
    rows = []
    for i, params in enumerate(GRID):
        m, p = fit(params, X["train"], y["train"], X["val"], y["val"])
        s_val = ins.model_scores("lightgbm", m, X["val"])
        rows.append({"params": p, "val_pr_auc": ins._pr_auc(y["val"], s_val), "_m": m})
        if i % 12 == 0:
            print(f"{i + 1}/{len(GRID)} best val PR-AUC so far "
                  f"{max(r['val_pr_auc'] for r in rows):.4f}", flush=True)  # fmt: skip
    best = max(rows, key=lambda r: r["val_pr_auc"])
    m = best["_m"]
    thr = ins.choose_threshold(y["val"], ins.model_scores("lightgbm", m, X["val"]))
    chal = {k: metrics(m, thr, k) for k in ("val", "test")}
    keep = (chal["test"]["precision"] >= champion["test"]["precision"]
            and chal["test"]["recall"] >= champion["test"]["recall"])  # fmt: skip
    # unseen users (seed 0) for the chosen setting vs the deployed setting
    insiders = set(df.loc[df["insider"] == 1, "user"])
    us = ins.user_split(df["user"], insiders, 0)
    up = {k: df[us == k] for k in ("train", "val", "test")}
    UX = {k: v[cols].to_numpy(np.float32) for k, v in up.items()}
    Uy = {k: v["label"].to_numpy() for k, v in up.items()}
    users: dict[str, Any] = {}
    for name, params in (("deployed_setting", {k: v for k, v in champ["params"].items() if k != "n_estimators"}),
                         ("chosen_setting", {k: v for k, v in best["params"].items() if k != "n_estimators"})):  # fmt: skip
        um, _ = fit(params, UX["train"], Uy["train"], UX["val"], Uy["val"])
        uthr = ins.choose_threshold(
            Uy["val"], ins.model_scores("lightgbm", um, UX["val"])
        )
        users[name] = ins.point_metrics(Uy["test"], ins.model_scores("lightgbm", um, UX["test"]),
                                        uthr, int(up["test"]["day"].nunique()))  # fmt: skip
    doc: dict[str, Any] = {
        "what": "insider LightGBM grid on CERT r4.2 time split (same split as deployed)",
        "grid_size": len(GRID), "chosen_params": best["params"], "chosen_val_pr_auc": best["val_pr_auc"],
        "rule": "choose on validation PR-AUC, threshold = validation F1 max; replace only if "
                "test precision AND recall not worse",
        "champion": champion, "challenger": chal, "unseen_users_seed0": users,
        "decision": "promoted" if keep else "kept champion", "saved": False,
        "seconds": round(time.time() - t0),
    }  # fmt: skip
    if keep and not a.no_save:
        bundle = ins.make_bundle(
            "lightgbm", m, best["params"], cols, X["val"], y["val"], X["train"]
        )
        bundle["threshold"] = thr
        bundle["version"] = int(champ.get("version", 1)) + 1
        ins.save_bundle(bundle)
        doc["saved"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "improve_insider.json").write_text(
        json.dumps(doc, indent=1, default=float), "utf-8"
    )
    print(json.dumps({"champion_test": champion["test"], "challenger_test": chal["test"],
                      "unseen_users": users, "decision": doc["decision"]}, indent=1, default=float))  # fmt: skip


if __name__ == "__main__":
    main()
