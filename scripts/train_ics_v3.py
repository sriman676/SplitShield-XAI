"""ICS v3 (owner 2026-10-22): more ICS data + a model that learns what attacks look like.

Data (owner-approved downloads, 2026-10-22):
  * HAI 22.04 (already used) and HAI 23.05 (github.com/icsdataset/hai), same plant;
    seven setpoint tags were renamed in 23.05 (matched by value range, COLUMN_MAP);
  * HAIEnd 23.05: the same 23.05 run with the boiler DCS's internal tags (--haiend);
  * BATADAL (scy-phy/ICS-Evasion-Attacks): a different plant, water distribution
    (--batadal).
HAI 21.03 stays the untouched blind set (never read here).

Splits (time order kept; every file belongs to one split):
  normal train   22.04 train1-5 and 23.05 train1-4, first 85 % of each version
  normal val     the last 15 % of each version (false-alarm budget)
  attack train   22.04 test1, 23.05 test1 (labelled)
  attack val     22.04 test2 (model choice, early stopping, threshold)
  test A         22.04 test3 + test4: the deployed model's test -> the rule decides here
  test B         23.05 test2: a newer plant run, reported

Candidates, each with 3 seeds:
  A  autoencoder on the normal data of both versions (normal-only, as deployed now)
  B  hybrid: that autoencoder's per-sensor errors + trend features -> LightGBM trained
     on normal rows and the labelled attack-train files (src/mlareas/ics_hybrid.py)
Choice on attack-val F1 (<= 5 false alarm events/day on normal val and on the normal
stretches of attack val). It replaces the deployed ICS v2 only if on test A precision
AND recall are not lower, for the seed-0 model (the one deployed) and the 3-seed mean.

    python -m scripts.train_ics_v3 [--no-save] [--seeds 0 1 2]
    python -m scripts.train_ics_v3 --haiend      (report only)
    python -m scripts.train_ics_v3 --batadal     (report only)
Results: results/extra/ics_v3.{json,md}, ics_v3_haiend.json, ics_v3_batadal.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.mlareas import common, ics, ics_hybrid
from utils.paths import PROJECT_ROOT, RESULTS_DIR

ICS_DATA = Path("D:/SplitShield-data/ics")
OUT = RESULTS_DIR / "extra"
# HAI 23.05 name -> HAI 22.04 name (same min / median / max; P1_B4005 has no match)
COLUMN_MAP = {
    "x1001_05_SETPOINT_OUT": "P1_B2004", "x1001_15_ASSIGN_OUT": "P1_B2016",
    "x1002_07_SETPOINT_OUT": "P1_B3004", "x1002_08_SETPOINT_OUT": "P1_B3005",
    "x1003_18_SETPOINT_OUT": "P1_B4002", "x1003_10_SETPOINT_OUT": "P1_B400B",
    "x1003_24_SUM_OUT": "P1_B4022",
}  # fmt: skip
GBM = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
       "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
       "verbose": -1, "num_threads": 4, "metric": "average_precision", "first_metric_only": True}  # fmt: skip
# early stopping on average precision: with attack rows up-weighted, validation log-loss
# rises from the first trees (2026-10-22: stopped at 1-5 trees), AP measures the ranking
OWNER_EXCEPTION = (
    "2026-10-22: owner chose candidate A (autoencoder on HAI 22.04 + 23.05) although its "
    "22.04 test recall is lower (0.142 vs 0.167, 14 vs 15 of 34 attacks, precision 0.893 vs "
    "0.895), because ICS v2 alarms almost nonstop on the newer HAI 23.05 run (precision "
    "0.04) while A reaches precision 0.70 / recall 0.22 there; v2 backed up for rollback"
)
NORMAL_EVERY = 10  # every 10th normal second goes into the classifier's training rows
GRID = np.unique(np.concatenate([np.linspace(0.02, 0.98, 49), [0.99, 0.995, 0.999]]))


def _read(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p).rename(columns=COLUMN_MAP)
    if "Attack" not in df.columns:
        try:
            df["Attack"] = ics._labels_for(p, df)
        except ValueError:
            df["Attack"] = 0
    return df


def load_frames() -> dict[str, Any]:
    """Every file in time order, keyed by its split."""
    d22 = common.raw_data_dir("hai")
    d23 = ICS_DATA / "hai-23.05"
    t22 = [_read(p) for p in sorted(d22.glob("hai-22.04_train*.csv"))]
    a22 = [_read(p) for p in sorted(d22.glob("hai-22.04_test*.csv"))]
    t23 = [_read(p) for p in sorted(d23.glob("hai-train*.csv"))]
    a23 = [_read(p) for p in sorted(d23.glob("hai-test*.csv"))]
    if len(a22) != 4 or len(a23) != 2 or not t22 or not t23:
        raise FileNotFoundError("HAI 22.04 (4 test files) and 23.05 (2 test files) are needed")
    return {"t22": t22, "t23": t23, "a22": a22, "a23": a23}


def _normal_split(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """First 85 % of a version's normal time -> train, the rest -> validation."""
    total = sum(len(f) for f in frames)
    cut, tr, va, start = int(0.85 * total), [], [], 0
    for f in frames:
        a, b = start, start + len(f)
        if b <= cut:
            tr.append(f)
        elif a >= cut:
            va.append(f)
        else:
            tr.append(f.iloc[: cut - a])
            va.append(f.iloc[cut - a :])
        start = b
    return [f for f in tr if len(f) > ics.LOOKBACK], [f for f in va if len(f) > ics.LOOKBACK]


def build(fr: dict[str, Any]) -> dict[str, Any]:
    tr22, va22 = _normal_split(fr["t22"])
    tr23, va23 = _normal_split(fr["t23"])
    sets = {"normal_train": tr22 + tr23, "normal_val": va22 + va23,
            "attack_train": [fr["a22"][0], fr["a23"][0]], "attack_val": [fr["a22"][1]],
            "test_A": fr["a22"][2:], "test_B": [fr["a23"][1]]}  # fmt: skip
    shared = set.intersection(*(set(f.columns) for v in sets.values() for f in v))
    normal = pd.concat([f[sorted(shared)] for f in sets["normal_train"]], ignore_index=True)
    prep = ics.fit_prep(normal)
    del normal
    out: dict[str, Any] = {"prep": prep, "sets": {}}
    for k, frames in sets.items():
        arr = np.vstack([prep.transform(f) for f in frames])
        y = np.concatenate([(f["Attack"].to_numpy() > 0) for f in frames]).astype(np.int8)
        out["sets"][k] = {"arr": arr, "y": y, "lengths": [len(f) for f in frames]}
    out["dropped"] = sorted(set().union(*(set(f.columns) for v in sets.values() for f in v)) - shared)
    return out


# ------------------------------------------------------------ evaluation


def _evaluate(score: np.ndarray, y: np.ndarray, thr: float) -> dict[str, Any]:
    a = (score > thr).astype(int)
    return ics.point_metrics(a, y) | ics.event_metrics(a, y)


def _choose(val_s: np.ndarray, val_y: np.ndarray, clean: np.ndarray,
            grid: np.ndarray) -> tuple[float, dict[str, Any]]:  # fmt: skip
    """Highest attack-val F1 within the alarm budget (normal val and val's normal parts)."""
    days = max(len(clean) / 86_400, 1e-9)
    best: tuple[float, float] = (-1.0, float(np.max(clean)) + 1e-9)
    for t in grid:
        if len(ics.alarm_events((clean > t).astype(int))) / days > ics.ALARM_BUDGET:
            continue
        m = _evaluate(val_s, val_y, float(t))
        if m["false_alarms_per_day"] > ics.ALARM_BUDGET:
            continue
        if m["f1"] > best[0]:
            best = (m["f1"], float(t))
    return best[1], _evaluate(val_s, val_y, best[1])


def _short(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("precision", "recall", "f1", "episodes", "episodes_detected", "false_alarms_per_day",
            "delay_s_median")  # fmt: skip
    return {k: (round(m[k], 4) if isinstance(m.get(k), float) else m.get(k)) for k in keys}


def _ae(D: dict[str, Any], seed: int) -> Any:
    ics.torch.set_num_threads(4)
    s = D["sets"]
    det = ics.fit_detector("autoencoder", s["normal_train"]["arr"],
                           ics.valid_index(s["normal_train"]["lengths"]), seed,
                           epochs=5, per_epoch=200_000)  # fmt: skip
    nv = s["normal_val"]
    return ics.rescale_per_sensor(det, nv["arr"], ics.valid_index(nv["lengths"]), 0.99, seed)


def _ae_scores(det: Any, part: dict[str, Any]) -> np.ndarray:
    return ics.full_scores(det, part["arr"], part["lengths"])


def _features(det: Any, part: dict[str, Any], every: int = 1, agg: bool = False) -> np.ndarray:
    """Feature rows of one split (computed per file, then thinned to every ``every``th);
    ``agg`` -> the sensor-agnostic aggregate (ics_hybrid.aggregate)."""
    out, start = [], 0
    for n in part["lengths"]:
        a = part["arr"][start : start + n]
        dev = ics_hybrid.sensor_dev(det, a, [n])
        X = ics_hybrid.features(a, dev, [n])[::every]
        out.append(ics_hybrid.aggregate(X, a.shape[1]) if agg else X)
        start += n
    return np.vstack(out)


def _thin_y(part: dict[str, Any], every: int) -> np.ndarray:
    out, start = [], 0
    for n in part["lengths"]:
        out.append(part["y"][start : start + n][::every])
        start += n
    return np.concatenate(out)


def run_seed(D: dict[str, Any], seed: int) -> dict[str, Any]:

    s = D["sets"]
    t0 = time.time()
    det = _ae(D, seed)
    res: dict[str, Any] = {"seed": seed}
    # --- A: autoencoder alone (normal-only)
    sc = {k: _ae_scores(det, s[k]) for k in ("normal_val", "attack_val", "attack_train", "test_A", "test_B")}
    grid_a = np.quantile(sc["normal_val"], [0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.0])
    thr_a, val_a = _choose(sc["attack_val"], s["attack_val"]["y"], sc["normal_val"], grid_a)
    res["A"] = {"threshold": thr_a, "train": _short(_evaluate(sc["attack_train"], s["attack_train"]["y"], thr_a)),
                "val": _short(val_a)} | {k: _short(_evaluate(sc[k], s[k]["y"], thr_a)) for k in ("test_A", "test_B")}  # fmt: skip
    for cand, agg in (("B", False), ("C", True)):
        res[cand], res[f"_booster_{cand}"] = _hybrid(det, s, seed, agg)
    res["seconds"] = round(time.time() - t0)
    res["_det"], res["_thr_a"] = det, thr_a
    return res


def _hybrid(det: Any, s: dict[str, Any], seed: int, agg: bool) -> tuple[dict[str, Any], Any]:
    """B (per-sensor features) or C (sensor-agnostic aggregate) -> LightGBM."""
    import lightgbm as lgb

    Xn = _features(det, s["normal_train"], NORMAL_EVERY, agg)
    Xa = _features(det, s["attack_train"], 1, agg)
    X = np.vstack([Xn, Xa])
    y = np.concatenate([np.zeros(len(Xn), np.int8), s["attack_train"]["y"]])
    del Xn
    Xv = _features(det, s["attack_val"], 1, agg)
    pos = max(1, int(y.sum()))
    params = GBM | {"seed": seed, "scale_pos_weight": min(20.0, (len(y) - pos) / pos)}
    booster = lgb.train(params, lgb.Dataset(X, y), 3000,
                        valid_sets=[lgb.Dataset(Xv, s["attack_val"]["y"])],
                        callbacks=[lgb.early_stopping(100, verbose=False)])  # fmt: skip

    def prob(Xp: np.ndarray, part: dict[str, Any]) -> np.ndarray:
        p = np.asarray(booster.predict(Xp, num_iteration=booster.best_iteration))
        return ics_hybrid.smooth(p, part["lengths"])

    p_train = prob(Xa, s["attack_train"])
    p_val = prob(Xv, s["attack_val"])
    p_clean = prob(_features(det, s["normal_val"], 1, agg), s["normal_val"])
    thr_b, val_b = _choose(p_val, s["attack_val"]["y"], p_clean, GRID)
    out = {"threshold": thr_b, "best_iteration": booster.best_iteration,
           "train": _short(_evaluate(p_train, s["attack_train"]["y"], thr_b)), "val": _short(val_b)}  # fmt: skip
    for k in ("test_A", "test_B"):
        out[k] = _short(_evaluate(prob(_features(det, s[k], 1, agg), s[k]), s[k]["y"], thr_b))
    return out, booster


def champion(D: dict[str, Any]) -> dict[str, Any]:
    """The deployed ICS v2 on the same tests (its own sensors, its own threshold)."""
    b = ics.load(refresh=True)
    fr = D["_frames"]
    out: dict[str, Any] = {"model": f"{b['family']} v{b.get('version', 1)}", "threshold": float(b["threshold"])}
    for k, frames in (("test_A", fr["a22"][2:]), ("test_B", [fr["a23"][1]])):
        arrs, ys, missing = [], [], set()
        for f in frames:
            f = f.copy()
            for i, c in enumerate(b["prep"].columns):
                if c not in f.columns:
                    missing.add(c)
                    f[c] = b["prep"].lo[i] + b["prep"].span[i] / 2
            arrs.append(b["prep"].transform(f))
            ys.append((f["Attack"].to_numpy() > 0).astype(np.int8))
        sc = ics.bundle_scores(b, np.vstack(arrs), [len(a) for a in arrs])
        out[k] = _short(_evaluate(sc, np.concatenate(ys), float(b["threshold"])))
        out[k]["missing_sensors_filled"] = sorted(missing)
    return out


def _mean(rows: list[dict[str, Any]], cand: str, split: str) -> dict[str, float]:
    return {m: round(float(np.mean([r[cand][split][m] for r in rows])), 4)
            for m in ("precision", "recall", "f1", "episodes_detected")}  # fmt: skip


def main_hai(seeds: list[int], save: bool, force: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    fr = load_frames()
    D = build(fr)
    D["_frames"] = fr
    champ = champion(D)
    rows = [run_seed(D, s) for s in seeds]
    cands = ("A", "B", "C")
    summary = {c: {sp: _mean(rows, c, sp) for sp in ("train", "val", "test_A", "test_B")} for c in cands}
    pick = force or max(cands, key=lambda c: summary[c]["val"]["f1"])
    r0 = rows[0][pick]["test_A"]
    m = summary[pick]["test_A"]
    c = champ["test_A"]
    better = (r0["precision"] >= c["precision"] and r0["recall"] >= c["recall"]
              and m["precision"] >= c["precision"] and m["recall"] >= c["recall"])  # fmt: skip
    doc: dict[str, Any] = {
        "what": "ICS v3: HAI 22.04 + 23.05, normal-only autoencoder (A) vs hybrid autoencoder + LightGBM "
                "trained with labelled attacks, per-sensor features (B) or sensor-agnostic aggregates (C)",
        "splits": {"normal_train": "22.04 train1-5 + 23.05 train1-4, first 85 % each",
                   "normal_val": "last 15 % of each version", "attack_train": "22.04 test1, 23.05 test1",
                   "attack_val": "22.04 test2", "test_A": "22.04 test3+4 (decides)", "test_B": "23.05 test2"},
        "rows": {k: len(v["y"]) for k, v in D["sets"].items()},
        "attack_rows": {k: int(v["y"].sum()) for k, v in D["sets"].items()},
        "sensors": len(D["prep"].columns), "dropped_columns": D["dropped"], "column_map": COLUMN_MAP,
        "blind_used": False,
        "rule": "pick A or B on attack-val F1 (alarm budget 5/day); replace the deployed ICS v2 only if "
                "test A precision AND recall are not lower, for the seed-0 model and the 3-seed mean",
        "champion": champ, "per_seed": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "mean": summary, "picked_on_validation": pick,
        "fit": {cand: common.fit_verdict(summary[cand]["train"]["f1"], summary[cand]["val"]["f1"],
                                         tree_model=cand != "A") for cand in cands},
    }  # fmt: skip
    if force:  # owner decision, recorded with the numbers it was made on
        better = True
        doc["owner_exception"] = OWNER_EXCEPTION
    if better:
        doc["decision"] = "promoted" + (f" (owner exception: {force})" if force else "")
        if save:
            _save(D, rows[0], pick)
            doc["saved_version"] = 3
    else:
        doc["decision"] = (f"kept ICS v2: {pick} test A seed-0 P/R {r0['precision']:.3f}/{r0['recall']:.3f}, "
                           f"mean {m['precision']:.3f}/{m['recall']:.3f} vs v2 {c['precision']:.3f}/"
                           f"{c['recall']:.3f}")  # fmt: skip
    doc["seconds"] = round(time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    if force:  # the 3-seed comparison (ics_v3.json / .md) stays as it is
        (OUT / "ics_v3_deploy.json").write_text(json.dumps(doc, indent=1, default=str), "utf-8")
        return doc
    (OUT / "ics_v3.json").write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    _write_md(doc)
    return doc


def _save(D: dict[str, Any], r: dict[str, Any], pick: str) -> None:
    cur = common.area_dir(ics.AREA) / "current"
    if cur.exists():  # keep a restorable copy of the deployed ICS v2
        dst = PROJECT_ROOT / "artifacts" / "deploy_backup" / f"{time.strftime('%Y%m%d-%H%M')}_pre_ics_v3"
        shutil.copytree(cur, dst)
    det = r["_det"]
    nv = D["sets"]["normal_val"]
    b = {"area": ics.AREA, "dataset": "HAI 22.04 + HAI 23.05", "seed": r["seed"], "version": 3,
         "family": ics.HYBRID if pick in ("B", "C") else "autoencoder",
         "model": ics._export_model(det), "prep": D["prep"], "norm": det.norm, "mean": det.mean,
         "threshold": r[pick]["threshold"], "ae_threshold": r["_thr_a"], "column_map": COLUMN_MAP,
         "percentile": None, "threshold_basis": "attack-val F1 within 5 false alarms/day (22.04 test2)",
         "val_clean": nv["arr"][:60_000], "val_injected": nv["arr"][:0], "val_labels": nv["y"][:0],
         "replay": D["sets"]["normal_train"]["arr"][-60_000:]}  # fmt: skip
    if pick in ("B", "C"):
        bst = r[f"_booster_{pick}"]
        b["gbm"] = bst.model_to_string(num_iteration=bst.best_iteration)
        b["feature_mode"] = "aggregate" if pick == "C" else "per_sensor"
        b["features"] = (ics_hybrid.agg_names() if pick == "C"
                         else ics_hybrid.feature_names(D["prep"].columns))  # fmt: skip
    common.save_bundle(ics.AREA, b)
    ics.load(refresh=True)


def _write_md(doc: dict[str, Any]) -> None:
    c = doc["champion"]
    lines = ["# 🏭 ICS v3: more data + attack-aware model", "",
             f"Sensors: {doc['sensors']}; rows {doc['rows']}; attack rows {doc['attack_rows']}.",
             f"Blind set (HAI 21.03) used: {doc['blind_used']}.", "",
             "| Model | Split | Precision | Recall | F1 | Attacks found | False alarms/day |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for k in ("test_A", "test_B"):
        x = c[k]
        lines.append(f"| deployed v2 ({c['model']}) | {k} | {x['precision']} | {x['recall']} | {x['f1']} | "
                     f"{x['episodes_detected']} / {x['episodes']} | {x['false_alarms_per_day']} |")  # fmt: skip
    r0 = doc["per_seed"][0]
    for cand, name in (("A", "A: autoencoder, normal only"), ("B", "B: hybrid, per-sensor features"),
                       ("C", "C: hybrid, sensor-agnostic features")):  # fmt: skip
        for k in ("train", "val", "test_A", "test_B"):
            x = r0[cand][k]
            lines.append(f"| {name} (seed 0) | {k} | {x['precision']} | {x['recall']} | {x['f1']} | "
                         f"{x['episodes_detected']} / {x['episodes']} | {x['false_alarms_per_day']} |")  # fmt: skip
    lines += ["", "3-seed means: " + json.dumps(doc["mean"]), "",
              f"Picked on validation: **{doc['picked_on_validation']}**. Decision: **{doc['decision']}**.",
              f"Fit: {json.dumps(doc['fit'])}", "", f"Rule: {doc['rule']}."]  # fmt: skip
    (OUT / "ics_v3.md").write_text("\n".join(lines), "utf-8")


# ------------------------------------------------------------ report-only experiments


def _supervised(normal: list[np.ndarray], clean: np.ndarray, att_tr: np.ndarray, y_tr: np.ndarray,
                att_va: np.ndarray, y_va: np.ndarray, att_te: np.ndarray, y_te: np.ndarray,
                windows: tuple[int, int, int], days_per_row: float, every: int,
                seed: int = 0) -> dict[str, Any]:  # fmt: skip
    """Trend features (no autoencoder) + LightGBM; threshold on validation within the
    alarm budget measured on ``clean`` (normal rows kept out of training). Features are
    computed per normal segment at full rate, then every ``every``-th row is kept."""
    import lightgbm as lgb

    # no autoencoder here: its slot carries the (smoothed) sensor level instead
    f = lambda a: ics_hybrid.features(a, a, [len(a)], windows)  # noqa: E731
    Xn = np.vstack([f(a)[::every] for a in normal])
    Xt, Xv, Xe, Xc = f(att_tr), f(att_va), f(att_te), f(clean)
    X = np.vstack([Xn, Xt])
    y = np.concatenate([np.zeros(len(Xn), np.int8), y_tr])
    pos = max(1, int(y.sum()))
    bst = lgb.train(GBM | {"seed": seed, "scale_pos_weight": min(20.0, (len(y) - pos) / pos),
                           "min_data_in_leaf": 20},
                    lgb.Dataset(X, y), 2000, valid_sets=[lgb.Dataset(Xv, y_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])  # fmt: skip
    pr = lambda Z: ics_hybrid.smooth(np.asarray(bst.predict(Z, num_iteration=bst.best_iteration)), [len(Z)])  # noqa: E731
    clean = pr(Xc)
    budget_days = len(clean) * days_per_row
    best = (-1.0, 0.5)
    for t in GRID:
        fa = len(ics.alarm_events((clean > t).astype(int))) / max(budget_days, 1e-9)
        if fa > ics.ALARM_BUDGET:
            continue
        f1 = ics.point_metrics((pr(Xv) > t).astype(int), y_va)["f1"]
        if f1 > best[0]:
            best = (f1, float(t))
    thr = best[1]
    ev = lambda Z, yy: ics.point_metrics((pr(Z) > thr).astype(int), yy)  # noqa: E731
    return {"threshold": thr, "train": ev(Xt, y_tr), "val": ev(Xv, y_va), "test": ev(Xe, y_te),
            "test_episodes": ics.event_metrics((pr(Xe) > thr).astype(int), y_te)}  # fmt: skip


def batadal() -> dict[str, Any]:
    """Second plant (water distribution, 15-minute samples). Train normal = training set;
    labelled test set 1 split in time (first 60 % attack-train, rest validation);
    test set 2 = test. Windows in samples: 4 (1 h), 24 (6 h), 96 (24 h)."""
    d = ICS_DATA / "batadal"
    rd = lambda n: pd.read_csv(d / n)  # noqa: E731
    tr, t1, t2 = rd("train_dataset.csv"), rd("test_dataset_1.csv"), rd("test_dataset_2.csv")
    cols = [c for c in tr.columns if c not in ("DATETIME", "ATT_FLAG", "Unnamed: 0") and tr[c].nunique() > 1]
    lo, span = tr[cols].min().to_numpy(), (tr[cols].max() - tr[cols].min()).to_numpy() + 1e-6
    nrm = lambda df: np.clip((df[cols].to_numpy(np.float32) - lo) / span, -10, 10).astype(np.float32)  # noqa: E731
    cut, nc = int(0.6 * len(t1)), int(0.85 * len(tr))
    y1, y2 = (t1["ATT_FLAG"].to_numpy() > 0).astype(np.int8), (t2["ATT_FLAG"].to_numpy() > 0).astype(np.int8)
    rows = [_supervised([nrm(tr.iloc[:nc])], nrm(tr.iloc[nc:]), nrm(t1.iloc[:cut]), y1[:cut],
                        nrm(t1.iloc[cut:]), y1[cut:], nrm(t2), y2, (4, 24, 96), 1 / 96, 1, seed)
            for seed in (0, 1, 2)]  # fmt: skip
    doc = {"what": "BATADAL water network (second plant), trend features + LightGBM, report only",
           "sensors": len(cols), "rows": {"normal": len(tr), "test1": len(t1), "test2": len(t2)},
           "attack_rows": {"test1": int(y1.sum()), "test2": int(y2.sum())}, "per_seed": rows,
           "test_mean": {m: round(float(np.mean([r["test"][m] for r in rows])), 4)
                         for m in ("precision", "recall", "f1")}}  # fmt: skip
    (OUT / "ics_v3_batadal.json").write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    return doc


def haiend() -> dict[str, Any]:
    """Do the boiler DCS's internal tags (HAIEnd 23.05) help? Same 23.05 run with and without
    them: normal = 23.05 train (first 85 %), attack-train = test1, test2 first half =
    validation, second half = test. Report only (a plant rarely exports these tags)."""
    d23, de = ICS_DATA / "hai-23.05", ICS_DATA / "haiend-23.05"
    out: dict[str, Any] = {"what": "HAIEnd 23.05 internal DCS tags vs HAI 23.05 tags only, report only"}
    for name, use_end in (("hai_only", False), ("hai_plus_haiend", True)):
        def load(kind: str, i: int, use_end: bool = use_end) -> pd.DataFrame:
            h = _read(d23 / f"hai-{kind}{i}.csv")
            if use_end:
                e = pd.read_csv(de / f"end-{kind}{i}.csv")
                e = e.select_dtypes("number").drop(columns=[c for c in ("Attack", "label") if c in e.columns])
                h = pd.concat([h.reset_index(drop=True), e.add_prefix("END_").reset_index(drop=True)], axis=1)
            return h

        segs = [load("train", i) for i in (1, 2, 3, 4)]
        nc = int(0.85 * len(segs[-1]))  # the end of train4 = clean rows for the alarm budget
        clean_df, segs[-1] = segs[-1].iloc[nc:], segs[-1].iloc[:nc]
        t1, t2 = load("test", 1), load("test", 2)
        both = pd.concat(segs, ignore_index=True)
        num = both.select_dtypes("number").columns
        cols = [c for c in num if c != "Attack" and both[c].nunique() > 1]
        lo = both[cols].min().to_numpy()
        span = (both[cols].max() - both[cols].min()).to_numpy() + 1e-6
        del both
        nrm = lambda df: np.clip((df[cols].to_numpy(np.float32) - lo) / span, -10, 10).astype(np.float32)  # noqa: E731,B023
        h = len(t2) // 2
        y1, y2 = t1["Attack"].to_numpy().astype(np.int8), t2["Attack"].to_numpy().astype(np.int8)
        out[name] = _supervised([nrm(x) for x in segs], nrm(clean_df), nrm(t1), y1, nrm(t2.iloc[:h]), y2[:h],
                                nrm(t2.iloc[h:]), y2[h:], ics_hybrid.WINDOWS, 1 / 86_400, NORMAL_EVERY)  # fmt: skip
        out[name]["sensors"] = len(cols)
        del segs
    (OUT / "ics_v3_haiend.json").write_text(json.dumps(out, indent=1, default=str), "utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--haiend", action="store_true")
    ap.add_argument("--batadal", action="store_true")
    ap.add_argument("--deploy", choices=["A", "B", "C"],
                    help="owner exception: save this candidate (seed = first --seeds) regardless of the rule")
    a = ap.parse_args()
    if a.batadal:
        res = batadal()
    elif a.haiend:
        res = haiend()
    else:
        res = main_hai(a.seeds[:1] if a.deploy else a.seeds, not a.no_save, a.deploy)
        res = {k: res[k] for k in ("decision", "picked_on_validation", "mean", "champion")}
    print(json.dumps(res, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
